import json
import re
import time
from typing import List, Optional

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


class FileAttachment(BaseModel):
    filename: str
    content_base64: str
    mime_type: str


class SolveRequest(BaseModel):
    prompt: str
    files: Optional[List[FileAttachment]] = None
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

create_travel_expense
  payload: { "description": string, "date": "YYYY-MM-DD", "amount": number,
             "category"?: string }

register_payment
  payload: { "invoiceId"?: number, "amount": number, "date": "YYYY-MM-DD",
             "paymentTypeId"?: number }

create_credit_note
  payload: { "invoiceId": number, "date": "YYYY-MM-DD" }

update_customer
  payload: { "name": string, "email"?: string, "phoneNumber"?: string }

delete_travel_expense
  payload: { "id"?: number, "description"?: string }

update_employee
  payload: { "firstName": string, "lastName": string, "email"?: string,
             "phoneNumber"?: string }

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
    return ask_claude_with_content([{"type": "text", "text": prompt}])


def ask_claude_with_content(content: list) -> dict:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
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
    body = {"name": payload.get("name", "Unknown Customer"), "isCustomer": True}
    for field in ("email", "phoneNumber", "organizationNumber"):
        if payload.get(field):
            body[field] = payload[field]
    tx_post(base_url, token, "/customer", body)


def do_create_supplier(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "Unknown Supplier"), "isSupplier": True}
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


def do_create_travel_expense(base_url: str, token: str, payload: dict) -> None:
    body = {
        "description": payload.get("description", "Travel expense"),
        "date": payload.get("date", "2025-03-20"),
        "costType": {"id": 1},
        "isCompleted": False,
    }
    if payload.get("amount") is not None:
        body["amount"] = payload["amount"]
    tx_post(base_url, token, "/travelExpense", body)


def do_register_payment(base_url: str, token: str, payload: dict) -> None:
    # Find the invoice if not given directly
    invoice_id = payload.get("invoiceId")
    amount = payload.get("amount", 0)
    date = payload.get("date", "2025-03-20")
    payment_type_id = payload.get("paymentTypeId", 1)
    if not invoice_id:
        r = tx_get(base_url, token, "/invoice", {"invoiceDateFrom": "2020-01-01",
                                                   "invoiceDateTo": "2030-01-01",
                                                   "count": 1})
        if r.status_code == 200:
            invoices = r.json().get("values", [])
            if invoices:
                invoice_id = invoices[0]["id"]
    if not invoice_id:
        print("No invoice found to register payment against")
        return
    tx_post(base_url, token, f"/invoice/{invoice_id}/:payment", {
        "paymentDate": date,
        "paymentTypeId": payment_type_id,
        "paidAmount": amount,
    })


def do_create_credit_note(base_url: str, token: str, payload: dict) -> None:
    invoice_id = payload.get("invoiceId")
    date = payload.get("date", "2025-03-20")
    if not invoice_id:
        print("No invoiceId provided for credit note")
        return
    tx_post(base_url, token, f"/invoice/{invoice_id}/:createCreditNote", {
        "date": date,
    })


def do_update_customer(base_url: str, token: str, payload: dict) -> None:
    name = payload.get("name", "")
    r = tx_get(base_url, token, "/customer", {"name": name, "count": 1})
    if r.status_code != 200:
        print("Could not find customer to update")
        return
    customers = r.json().get("values", [])
    if not customers:
        print(f"No customer found with name: {name}")
        return
    customer_id = customers[0]["id"]
    update_body = {}
    for field in ("email", "phoneNumber", "organizationNumber"):
        if payload.get(field):
            update_body[field] = payload[field]
    if not update_body:
        print("No fields to update")
        return
    url = f"{base_url.rstrip('/')}/customer/{customer_id}"
    r2 = requests.put(url, auth=tx_auth(token), json=update_body, timeout=30)
    print(f"PUT /customer/{customer_id} -> {r2.status_code}: {r2.text[:200]}")


def do_delete_travel_expense(base_url: str, token: str, payload: dict) -> None:
    expense_id = payload.get("id")
    if not expense_id:
        r = tx_get(base_url, token, "/travelExpense", {"count": 1})
        if r.status_code == 200:
            expenses = r.json().get("values", [])
            if expenses:
                expense_id = expenses[0]["id"]
    if not expense_id:
        print("No travel expense found to delete")
        return
    url = f"{base_url.rstrip('/')}/travelExpense/{expense_id}"
    r2 = requests.delete(url, auth=tx_auth(token), timeout=30)
    print(f"DELETE /travelExpense/{expense_id} -> {r2.status_code}")


def do_update_employee(base_url: str, token: str, payload: dict) -> None:
    first = payload.get("firstName", "")
    last = payload.get("lastName", "")
    r = tx_get(base_url, token, "/employee",
               {"firstName": first, "lastName": last, "count": 1})
    if r.status_code != 200:
        return
    employees = r.json().get("values", [])
    if not employees:
        print(f"No employee found: {first} {last}")
        return
    employee_id = employees[0]["id"]
    update_body = {}
    for field in ("email", "phoneNumber"):
        if payload.get(field):
            update_body[field] = payload[field]
    if not update_body:
        return
    url = f"{base_url.rstrip('/')}/employee/{employee_id}"
    r2 = requests.put(url, auth=tx_auth(token), json=update_body, timeout=30)
    print(f"PUT /employee/{employee_id} -> {r2.status_code}: {r2.text[:200]}")


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
    "create_travel_expense": do_create_travel_expense,
    "register_payment": do_register_payment,
    "create_credit_note": do_create_credit_note,
    "update_customer": do_update_customer,
    "delete_travel_expense": do_delete_travel_expense,
    "update_employee": do_update_employee,
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
        user_content = []
        for f in (body.files or []):
            try:
                if f.mime_type == "application/pdf":
                    user_content.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": f.content_base64,
                        }
                    })
                elif f.mime_type.startswith("image/"):
                    user_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f.mime_type,
                            "data": f.content_base64,
                        }
                    })
            except Exception as e:
                print(f"Error processing file {f.filename}: {e}")
        user_content.append({"type": "text", "text": body.prompt})
        plan = ask_claude_with_content(user_content)
        print(f"Claude plan: {json.dumps(plan)}")
    except Exception as e:
        print(f"Claude error: {e} - falling back to no_op")
        return {"status": "completed"}
    for action in plan.get("actions", []):
        dispatch(base_url, token, action)
    return {"status": "completed"}
