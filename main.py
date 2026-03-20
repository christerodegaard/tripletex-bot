import re
from typing import Any, List, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Tripletex competition bot")


class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class SolveRequest(BaseModel):
    prompt: str
    files: Optional[List[Any]] = None
    tripletex_credentials: TripletexCredentials


def extract_quoted_name(prompt: str) -> Optional[str]:
    match = re.search(r'"([^"]*)"', prompt)
    if not match:
        return None
    return match.group(1).strip()


def print_post_result(action: str, r: requests.Response) -> None:
    print(f"action: {action}")
    print(f"response status code: {r.status_code}")
    print(f"response text: {r.text}")


def create_customer(base_url: str, token: str, name: str) -> None:
    base = base_url.rstrip("/")
    try:
        r = requests.post(
            f"{base}/customer",
            auth=("0", token),
            json={"name": name},
            timeout=30,
        )
        print_post_result("customer", r)
    except Exception as e:
        print(f"API error (customer): {e}")


def create_product(base_url: str, token: str, name: str) -> None:
    base = base_url.rstrip("/")
    try:
        r = requests.post(
            f"{base}/product",
            auth=("0", token),
            json={"name": name},
            timeout=30,
        )
        print_post_result("product", r)
    except Exception as e:
        print(f"API error (product): {e}")


def do_create_employee(base_url: str, token: str, payload: dict) -> None:
    first = payload.get("firstName", "")
    last = payload.get("lastName", "")

    # If Claude put the full name in firstName and left lastName empty, split it
    if first and not last:
        parts = first.rsplit(" ", 1)
        first = parts[0]
        last = parts[1] if len(parts) > 1 else "Unknown"

    body = {
        "firstName": first or "Unknown",
        "lastName": last or "Employee",
    }
    for field in ("email", "employeeNumber"):
        if payload.get(field):
            body[field] = payload[field]
    tx_post(base_url, token, "/employee", body)


def do_create_invoice(base_url: str, token: str, payload: dict) -> None:
    customer_name = payload.get("customer_name", "Unknown Customer")
    invoice_date = payload.get("invoiceDate", "2025-03-20")
    due_date = payload.get("invoiceDueDate", "2025-04-20")

    # Step 1: find or create the customer
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
        print("Could not find or create customer for invoice — aborting")
        return

    # Step 2: create order with order lines
    raw_orders = payload.get("orders") or []
    if not raw_orders:
        raw_orders = [{"description": "Service", "unitPriceExcludingVatCurrency": 0, "count": 1}]

    order_lines = [
        {
            "description": item.get("description", "Item"),
            "unitPriceExcludingVatCurrency": item.get("unitPriceExcludingVatCurrency", 0),
            "count": item.get("count", 1),
        }
        for item in raw_orders
    ]

    order_body = {
        "customer": {"id": customer_id},
        "orderDate": invoice_date,
        "orderLines": order_lines,
    }
    r3 = tx_post(base_url, token, "/order", order_body)
    if r3.status_code not in (200, 201):
        print(f"Order creation failed ({r3.status_code}) — cannot create invoice")
        return

    order_id = r3.json().get("value", {}).get("id")
    if not order_id:
        print("No order ID returned — cannot create invoice")
        return

    # Step 3: invoice from order
    invoice_body = {
        "invoiceDate": invoice_date,
        "invoiceDueDate": due_date,
        "orders": [{"id": order_id}],
    }
    tx_post(base_url, token, "/invoice", invoice_body)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/test-interpret")
def test_interpret(body: dict) -> dict:
    """Debug endpoint: returns Claude's action plan for a given prompt without calling Tripletex."""
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
    print(f"incoming prompt: {body.prompt}")

    base_url = body.tripletex_credentials.base_url
    token = body.tripletex_credentials.session_token
    prompt_lower = body.prompt.lower()
    quoted = extract_quoted_name(body.prompt)

    if "employee" in prompt_lower:
        print("employee not implemented yet")

    actions: List[str] = []
    if "create customer" in prompt_lower:
        actions.append("customer")
    if "create product" in prompt_lower:
        actions.append("product")

    if not actions:
        print("selected action: none")
        return {"status": "completed"}

    for action in actions:
        print(f"selected action: {action}")
        if action == "customer":
            create_customer(
                base_url,
                token,
                quoted if quoted is not None else "Auto Customer AS",
            )
        else:
            create_product(
                base_url,
                token,
                quoted if quoted is not None else "Auto Product",
            )

    return {"status": "completed"}
