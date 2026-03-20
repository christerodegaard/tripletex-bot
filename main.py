import json
import re
import time
from typing import Any, List, Optional

import anthropic
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Tripletex competition bot")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class SolveRequest(BaseModel):
    prompt: str
    files: Optional[List[Any]] = None
    tripletex_credentials: TripletexCredentials


# ---------------------------------------------------------------------------
# Claude: interpret prompt → structured action plan
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an accounting assistant that translates natural-language
instructions into Tripletex API actions.

Given a user prompt, respond ONLY with a JSON object (no markdown, no explanation).

The JSON must have this shape:
{
  "actions": [
    {
      "type": "<action_type>",
      "payload": { <fields needed for that action> }
    }
  ]
}

Supported action types and their payloads:

create_customer
  payload: { "name": string, "email"?: string, "phoneNumber"?: string,
             "organizationNumber"?: string }

create_supplier
  payload: { "name": string, "email"?: string, "phoneNumber"?: string,
             "organizationNumber"?: string }

create_product
  payload: { "name": string, "number"?: string, "costExcludingVatCurrency"?: number,
             "priceExcludingVatCurrency"?: number, "priceIncludingVatCurrency"?: number }

create_employee
  payload: { "firstName": string, "lastName": string, "email"?: string,
             "employeeNumber"?: string }

create_invoice
  payload: { "customer_name": string, "invoiceDate": "YYYY-MM-DD",
             "invoiceDueDate": "YYYY-MM-DD",
             "orders": [ { "description": string,
                           "unitPriceExcludingVatCurrency": number,
                           "count": number } ] }

create_project
  payload: { "name": string, "projectManagerEmail"?: string,
             "startDate"?: "YYYY-MM-DD", "endDate"?: "YYYY-MM-DD" }

create_department
  payload: { "name": string, "departmentNumber"?: string }

create_ledger_posting
  payload: { "description": string, "date": "YYYY-MM-DD",
             "debitAccount": string, "creditAccount": string, "amount": number }

no_op
  payload: { "reason": string }

Rules:
- If the prompt is ambiguous, do your best and pick the most likely action.
- If you truly cannot map it, use no_op.
- Extract names, dates, amounts from the prompt.
- For dates, use today (2025-03-20) if not specified.
- Return ONLY valid JSON, nothing else.
"""


def ask_claude(prompt: str) -> dict:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Tripletex HTTP helpers
# ---------------------------------------------------------------------------

def tx_auth(token: str):
    return ("0", token)


def tx_post(base_url: str, token: str, path: str, payload: dict) -> requests.Response:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    print(f"POST {url}  payload={json.dumps(payload)}")
    r = requests.post(url, auth=tx_auth(token), json=payload, timeout=30)
    print(f"  -> {r.status_code}: {r.text[:300]}")
    return r


def tx_get(base_url: str, token: str, path: str, params: dict = None) -> requests.Response:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    print(f"GET {url}  params={params}")
    r = requests.get(url, auth=tx_auth(token), params=params, timeout=30)
    print(f"  -> {r.status_code}: {r.text[:300]}")
    return r


# ---------------------------------------------------------------------------
# Action executors
# ---------------------------------------------------------------------------

def do_create_customer(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "Unknown Customer")}
    for field in ("email", "phoneNumber", "organizationNumber"):
        if payload.get(field):
            body[field] = payload[field]
    tx_post(base_url, token, "/customer", body)


def do_create_supplier(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "Unknown Supplier")}
    for field in ("email", "phoneNumber", "organizationNumber"):
        if payload.get(field):
            body[field] = payload[field]
    tx_post(base_url, token, "/customer", body)


def do_create_product(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "Unknown Product")}
    for field in ("number", "costExcludingVatCurrency",
                  "priceExcludingVatCurrency", "priceIncludingVatCurrency"):
        if payload.get(field) is not None:
            body[field] = payload[field]
    tx_post(base_url, token, "/product", body)


def do_create_employee(base_url: str, token: str, payload: dict) -> None:
    first = payload.get("firstName", "")
    last = payload.get("lastName", "")
    if first and not last:
        parts = first.rsplit(" ", 1)
        first = parts[0]
        last = parts[1] if len(parts) > 1 else "Unknown"
    body = {
        "firstName": first or "Unknown",
        "lastName": last or "Employee",
        "userType": "STANDARD",
    }
    for field in ("email", "employeeNumber"):
        if payload.get(field):
            body[field] = payload[field]
    if not body.get("email"):
        safe_name = (first + "." + last).lower().replace(" ", "")
        body["email"] = f"{safe_name}.{int(time.time())}@example.com"
    dept_r = tx_get(base_url, token, "/department", {"count": 1})
    if dept_r.status_code == 200:
        depts = dept_r.json().get("values", [])
        if depts:
            body["department"] = {"id": depts[0]["id"]}
    tx_post(base_url, token, "/employee", body)


def do_create_project(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "New Project")}
    if payload.get("startDate"):
        body["startDate"] = payload["startDate"]
    if payload.get("endDate"):
        body["endDate"] = payload["endDate"]
    if payload.get("projectManagerEmail"):
        r = tx_get(base_url, token, "/employee",
                   {"email": payload["projectManagerEmail"]})
        if r.status_code == 200:
            employees = r.json().get("values", [])
            if employees:
                body["projectManager"] = {"id": employees[0]["id"]}
    if "projectManager" not in body:
        r = tx_get(base_url, token, "/employee", {"count": 1})
        if r.status_code == 200:
            employees = r.json().get("values", [])
            if employees:
                body["projectManager"] = {"id": employees[0]["id"]}
    tx_post(base_url, token, "/project", body)


def do_create_department(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "New Department")}
    if payload.get("departmentNumber"):
        body["departmentNumber"] = payload["departmentNumber"]
    tx_post(base_url, token, "/department", body)


def do_create_invoice(base_url: str, token: str, payload: dict) -> None:
    customer_name = payload.get("customer_name", "Unknown Customer")
    invoice_date = payload.get("invoiceDate", "2025-03-20")
    due_date = payload.get("invoiceDueDate", "2025-04-20")

    # Step 1: find or create customer
    r = tx_get(base_url, token, "/customer", {"name": customer_name, "count": 1})
    customer_id = None
    if r.status_code == 200:
        customers = r.json().get("values", [])
        if customers:
            customer_id = customers[0]["id"]
    if customer_id is None:
        r2 = tx_post(base_url, token, "/customer", {"name": customer_name})
        if r2.status_code in (200, 201):
            customer_id = r2.json().get("value", {}).get("id")
    if customer_id is None:
        print("Could not find or create customer for invoice - aborting")
        return

    # Step 2: create order
    raw_orders = payload.get("orders") or []
    if not raw_orders:
        raw_orders = [{"description": "Service",
                       "unitPriceExcludingVatCurrency": 0, "count": 1}]
    order_lines = [
        {
            "description": item.get("description", "Item"),
            "unitPriceExcludingVatCurrency": item.get(
                "unitPriceExcludingVatCurrency", 0),
            "count": item.get("count", 1),
        }
        for item in raw_orders
    ]
    order_body = {
        "customer": {"id": customer_id},
        "orderDate": invoice_date,
        "deliveryDate": invoice_date,
        "orderLines": order_lines,
    }
    r3 = tx_post(base_url, token, "/order", order_body)
    if r3.status_code not in (200, 201):
        print(f"Order creation failed ({r3.status_code}) - cannot create invoice")
        return
    order_id = r3.json().get("value", {}).get("id")
    if not order_id:
        print("No order ID returned - cannot create invoice")
        return

    # Step 3: invoice from order
    tx_post(base_url, token, "/invoice", {
        "invoiceDate": invoice_date,
        "invoiceDueDate": due_date,
        "orders": [{"id": order_id}],
    })


def do_create_ledger_posting(base_url: str, token: str, payload: dict) -> None:
    body = {
        "description": payload.get("description", "Manual posting"),
        "date": payload.get("date", "2025-03-20"),
        "debitAccount": {"number": payload.get("debitAccount", "1500")},
        "creditAccount": {"number": payload.get("creditAccount", "4000")},
        "amount": payload.get("amount", 0),
    }
    tx_post(base_url, token, "/ledger/voucher", body)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

ACTION_MAP = {
    "create_customer": do_create_customer,
    "create_supplier": do_create_supplier,
    "create_product": do_create_product,
    "create_employee": do_create_employee,
    "create_project": do_create_project,
    "create_department": do_create_department,
    "create_invoice": do_create_invoice,
    "create_ledger_posting": do_create_ledger_posting,
}


def dispatch(base_url: str, token: str, action: dict) -> None:
    action_type = action.get("type", "no_op")
    payload = action.get("payload", {})
    print(f"Dispatching: {action_type}  payload={payload}")
    if action_type == "no_op":
        print(f"no_op reason: {payload.get('reason', '(none)')}")
        return
    handler = ACTION_MAP.get(action_type)
    if handler:
        try:
            handler(base_url, token, payload)
        except Exception as e:
            print(f"Error in {action_type}: {e}")
    else:
        print(f"Unknown action type: {action_type}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/test-interpret")
def test_interpret(body: dict) -> dict:
    """Debug: returns Claude's action plan for a prompt without calling Tripletex."""
    prompt = body.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}
    try:
        plan = ask_claude(prompt)
        return {"prompt": prompt, "plan": plan}
    except Exception as e:
        return {"error": str(e)}


@app.post("/solve")
def solve(body: SolveRequest) -> dict:
    print(f"=== incoming prompt: {body.prompt!r} ===")
    base_url = body.tripletex_credentials.base_url
    token = body.tripletex_credentials.session_token
    try:
        plan = ask_claude(body.prompt)
        print(f"Claude plan: {json.dumps(plan)}")
    except Exception as e:
        print(f"Claude error: {e} - falling back to no_op")
        return {"status": "completed"}
    for action in plan.get("actions", []):
        dispatch(base_url, token, action)
    return {"status": "completed"}
