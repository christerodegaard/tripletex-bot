import json
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import anthropic
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Tripletex competition bot")


def keep_alive():
    while True:
        time.sleep(600)  # 10 minutes
        try:
            requests.get("https://tripletex-bot.onrender.com/health", timeout=10)
            print("Keep-alive ping sent")
        except Exception:
            pass


threading.Thread(target=keep_alive, daemon=True).start()


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

Note: Prompts may come in Norwegian. Key Norwegian terms:
- ansatt/medarbeider = employee
- kunde = customer
- leverandør = supplier
- faktura = invoice
- produkt = product
- avdeling = department
- prosjekt = project
- reiseregning = travel expense
- betaling = payment
- kreditnota = credit note

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
             "organizationNumber"?: string, "address"?: string,
             "postalCode"?: string, "city"?: string, "country"?: string }

create_supplier
  payload: { "name": string, "email"?: string, "phoneNumber"?: string,
             "organizationNumber"?: string, "address"?: string,
             "postalCode"?: string, "city"?: string, "country"?: string }

create_product
  payload: { "name": string, "number"?: string,
             "priceExcludingVatCurrency"?: number,
             "costExcludingVatCurrency"?: number }
  NEVER include vatTypeId or any VAT-related fields — they cause validation errors.
  Ignore any VAT rate mentioned in the prompt.

create_employee
  payload: { "firstName": string, "lastName": string, "email"?: string,
             "employeeNumber"?: string, "dateOfBirth"?: "YYYY-MM-DD",
             "startDate"?: "YYYY-MM-DD" }
  Always extract dateOfBirth (convert "21. January 1999" → "1999-01-21") and startDate when mentioned.

create_invoice
  payload: { "customer_name": string, "invoiceDate": "YYYY-MM-DD",
             "invoiceDueDate": "YYYY-MM-DD",
             "orders": [ { "description": string,
                           "unitPriceExcludingVatCurrency": number,
                           "count": number, "vatRate"?: number } ] }
  Extract vatRate per line when mentioned: 25 = high rate, 15 = food/reduced, 0 = exempt.
  Maps to VAT types: 25 -> number 3, 15 -> 33, 0 -> no vatType on the line.

create_project
  payload: { "name": string, "projectManagerEmail"?: string,
             "startDate"?: "YYYY-MM-DD", "endDate"?: "YYYY-MM-DD",
             "customerName"?: string }

create_department
  payload: { "name": string, "departmentNumber"?: string }

create_ledger_posting
  payload: { "description": string, "date": "YYYY-MM-DD",
             "debitAccount": "NNNN", "creditAccount": "NNNN", "amount": number,
             "departmentId"?: number, "departmentName"?: string }
  When a ledger posting must be linked to a dimension value, pass the dimension value name as departmentName.

create_travel_expense
  payload: { "employeeEmail"?: string, "description": string,
             "date": "YYYY-MM-DD", "departureFrom"?: string, "destination"?: string }
  Always extract employeeEmail when an email is mentioned.
  Use "description" for the trip title/purpose only.
  Do NOT include amount, cost, category, or any cost breakdown fields.

register_payment
  payload: { "customer_name"?: string, "invoiceId"?: number,
             "amount": number, "date": "YYYY-MM-DD" }
  Incoming payments. For reversals, prefer reverse_payment with a positive
  amount, or use a negative amount here to trigger automated reversal.

reverse_payment
  payload: { "customer_name"?: string, "invoiceId"?: number,
             "amount": number, "date": "YYYY-MM-DD" }
  Bank reversals and returned payments. Same fields as register_payment; use
  the original positive payment amount — reversal logic runs in the backend.

create_credit_note
  payload: { "customerName": string, "date": "YYYY-MM-DD", "amount"?: number }
  Extract the invoice amount from the prompt if mentioned (e.g. "12000 NOK").

update_customer
  payload: { "name": string, "email"?: string, "phoneNumber"?: string }

delete_travel_expense
  payload: { "id"?: number, "description"?: string }

update_employee
  payload: { "firstName": string, "lastName": string, "email"?: string,
             "phoneNumber"?: string }

log_hours
  payload: { "employeeEmail": string, "projectName": string, "activityName"?: string,
             "date": "YYYY-MM-DD", "hours": number, "hourlyRate"?: number }

update_product
  payload: { "name": string, "priceExcludingVatCurrency"?: number,
             "priceIncludingVatCurrency"?: number, "costExcludingVatCurrency"?: number }

create_contact
  payload: { "firstName": string, "lastName": string, "email"?: string,
             "phoneNumber"?: string, "customerId"?: number, "customerName"?: string }

delete_customer
  payload: { "name": string }

delete_employee
  payload: { "firstName": string, "lastName": string }

create_product_with_price
  payload: { "name": string, "priceExcludingVatCurrency": number,
             "costExcludingVatCurrency"?: number, "number"?: string }

register_supplier_invoice
  payload: { "supplierName": string, "organizationNumber"?: string,
             "invoiceNumber"?: string, "amount": number, "vatPercent"?: number,
             "accountCode"?: string, "date": "YYYY-MM-DD" }

create_accounting_dimension
  payload: { "name": string, "values": string[] }
  Creates a dimension (department) and its values as sub-departments.
  description, date, debitAccount, creditAccount, and amount belong only on create_ledger_posting.
  ALWAYS follow with a separate create_ledger_posting action for any required posting.

create_payroll
  payload: { "employeeEmail": string, "baseSalary": number,
             "bonus"?: number, "date": "YYYY-MM-DD" }
  Use this for ANY request to run payroll, process salary, pay wages, or add bonuses.
  Extract the employee email, base salary amount, and any bonus amount.
  Always use this action — never use no_op for payroll requests.

no_op
  payload: { "reason": string }

Rules:
- If the prompt is ambiguous, do your best and pick the most likely action.
- If you truly cannot map it, use no_op.
- Extract names, dates, amounts from the prompt.
- For dates, use today (2026-03-20) if not specified.
- Return ONLY valid JSON, nothing else.
- For payroll/salary/lønn/lønnskjøring requests, ALWAYS use create_payroll, never no_op.
- For accounting dimension requests, ALWAYS use create_accounting_dimension, never no_op.
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


_thread_local = threading.local()


def _get_account_cache() -> Dict[Any, int]:
    if not hasattr(_thread_local, "cache"):
        _thread_local.cache = {}
    return _thread_local.cache


def get_vat_type_id_by_number(
    base_url: str, token: str, number_str: str
) -> Optional[int]:
    """Look up VAT type id by ledger vatType number (e.g. 3, 33, 6)."""
    cache = _get_account_cache()
    key = f"vattype_{number_str}"
    if key in cache:
        return cache[key]
    r = tx_get(base_url, token, "/ledger/vatType", {"count": 50})
    if r.status_code == 200:
        for vt in r.json().get("values", []):
            if str(vt.get("number", "")) == number_str:
                cache[key] = vt["id"]
                print(
                    f"Found VAT type {number_str} -> id {vt['id']} ({vt.get('name')})"
                )
                return vt["id"]
    return None


def get_account_id(
    base_url: str, token: str, number: int
) -> Optional[int]:
    cache = _get_account_cache()
    key = f"acct_{number}"
    if key in cache:
        return cache[key]
    r = tx_get(base_url, token, "/ledger/account", {"number": number, "count": 1})
    if r.status_code == 200:
        vals = r.json().get("values", [])
        if vals:
            aid = vals[0]["id"]
            cache[key] = aid
            print(f"Account {number} -> id {aid}")
            return aid
    print(f"WARNING: Could not find account {number}")
    return None


def make_posting(
    base_url: str,
    token: str,
    date: str,
    description: str,
    account_number: int,
    amount: float,
    department_id: Optional[int] = None,
    row: int = 1,
    vat_type_id: Optional[int] = None,
) -> dict:
    resolved = get_account_id(base_url, token, account_number)
    account_id = resolved if resolved is not None else account_number
    posting = {
        "row": row,
        "date": date,
        "description": description,
        "account": {"id": account_id},
        "amount": amount,
        "amountCurrency": amount,
    }
    if department_id is not None:
        posting["department"] = {"id": department_id}
    if vat_type_id is not None:
        posting["vatType"] = {"id": vat_type_id}
    return posting


def lookup_vat_type_mva3(base_url: str, token: str) -> Optional[int]:
    """Resolve VAT type id for mva-kode 3 (Utgående avgift, høy sats) by exact number."""
    return get_vat_type_id_by_number(base_url, token, "3")


def set_bank_account(base_url: str, token: str) -> bool:
    # Try employee/current to get company ID
    for path in ["/employee/current", "/employee/current/employmentId"]:
        r = tx_get(base_url, token, path)
        print(f"set_bank_account {path} -> {r.status_code}: {r.text[:150]}")
        if r.status_code == 200:
            data = r.json().get("value", {})
            # Try different paths to company ID
            company_id = None
            if isinstance(data, dict):
                company_id = (data.get("company", {}) or {}).get("id")
                if not company_id:
                    company_id = (data.get("department", {}) or {}).get("company", {}).get("id") if data.get("department") else None
            if company_id:
                for put_path in [f"/company/{company_id}", f"/settings/company"]:
                    put_url = f"{base_url.rstrip('/')}{put_path}"
                    payload = {"id": company_id, "bankAccountNumber": "15060126900"}
                    r2 = requests.put(put_url, auth=tx_auth(token), json=payload, timeout=30)
                    print(f"PUT {put_path} -> {r2.status_code}: {r2.text[:150]}")
                    if r2.status_code in (200, 201):
                        return True
    return False


# ---------------------------------------------------------------------------
# Action executors
# ---------------------------------------------------------------------------

def do_create_customer(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "Unknown Customer"),
            "isCustomer": True,
            "isSupplier": False}
    for field in ("email", "phoneNumber", "organizationNumber"):
        if payload.get(field):
            body[field] = payload[field]
    # Build address if any address fields present
    addr = {}
    if payload.get("address"):
        addr["addressLine1"] = payload["address"]
    if payload.get("postalCode"):
        addr["postalCode"] = payload["postalCode"]
    if payload.get("city"):
        addr["city"] = payload["city"]
    if payload.get("country"):
        addr["country"] = payload["country"]
    if addr:
        body["physicalAddress"] = addr
    tx_post(base_url, token, "/customer", body)


def do_create_supplier(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "Unknown Supplier"),
            "isSupplier": True,
            "isCustomer": True}
    for field in ("email", "phoneNumber", "organizationNumber"):
        if payload.get(field):
            body[field] = payload[field]
    # Build address if any address fields present
    addr = {}
    if payload.get("address"):
        addr["addressLine1"] = payload["address"]
    if payload.get("postalCode"):
        addr["postalCode"] = payload["postalCode"]
    if payload.get("city"):
        addr["city"] = payload["city"]
    if payload.get("country"):
        addr["country"] = payload["country"]
    if addr:
        body["physicalAddress"] = addr
    r = tx_post(base_url, token, "/customer", body)
    if r.status_code in (200, 201):
        val = r.json().get("value", {})
        sid = val.get("id")
        if sid:
            put_body = {
                k: v
                for k, v in val.items()
                if k not in ("url", "changes", "displayName")
            }
            put_body["isCustomer"] = False
            put_url = f"{base_url.rstrip('/')}/customer/{sid}"
            r2 = requests.put(put_url, auth=tx_auth(token), json=put_body, timeout=30)
            print(f"PUT supplier isCustomer=False -> {r2.status_code}: {r2.text[:100]}")


def do_create_product(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "Unknown Product")}
    for field in ("number", "costExcludingVatCurrency", "priceExcludingVatCurrency"):
        if payload.get(field) is not None:
            body[field] = payload[field]
    r = tx_post(base_url, token, "/product", body)
    if r.status_code == 422 and "er i bruk" in r.text:
        print("Product number in use, retrying without number")
        body.pop("number", None)
        r = tx_post(base_url, token, "/product", body)
    if r.status_code == 422 and ("allerede registrert" in r.text or "er i bruk" in r.text):
        print("Product name also in use, looking up existing product")
        r_lookup = tx_get(base_url, token, "/product", {"name": body.get("name", ""), "count": 1})
        if r_lookup.status_code == 200:
            products = r_lookup.json().get("values", [])
            if products:
                print(f"Found existing product id: {products[0]['id']}")
                return  # Product already exists, that's fine


def do_update_product(base_url: str, token: str, payload: dict) -> None:
    name = payload.get("name", "")
    r = tx_get(base_url, token, "/product", {"name": name, "count": 1})
    if r.status_code != 200:
        return
    products = r.json().get("values", [])
    if not products:
        print(f"No product found: {name}")
        return
    product_id = products[0]["id"]
    update_body = {}
    for field in ("priceExcludingVatCurrency", "priceIncludingVatCurrency",
                  "costExcludingVatCurrency", "number"):
        if payload.get(field) is not None:
            update_body[field] = payload[field]
    if not update_body:
        return
    url = f"{base_url.rstrip('/')}/product/{product_id}"
    r2 = requests.put(url, auth=tx_auth(token), json=update_body, timeout=30)
    print(f"PUT /product/{product_id} -> {r2.status_code}: {r2.text[:200]}")


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
    if payload.get("dateOfBirth"):
        body["dateOfBirth"] = payload["dateOfBirth"]
    if not body.get("email"):
        safe_name = (first + "." + last).lower().replace(" ", "")
        body["email"] = f"{safe_name}.{int(time.time())}@example.com"
    dept_r = tx_get(base_url, token, "/department", {"count": 1})
    if dept_r.status_code == 200:
        depts = dept_r.json().get("values", [])
        if depts:
            body["department"] = {"id": depts[0]["id"]}
    r = tx_post(base_url, token, "/employee", body)
    if r.status_code == 422 and "finnes allerede" in r.text:
        print("Employee email exists, looking up existing employee")
        r2 = tx_get(base_url, token, "/employee", {"email": body["email"]})
        if r2.status_code == 200:
            employees = r2.json().get("values", [])
            if employees:
                print(f"Found existing employee id: {employees[0]['id']}")
    if r.status_code in (200, 201) and payload.get("startDate"):
        emp_id = r.json().get("value", {}).get("id")
        if emp_id:
            r_emp = tx_post(
                base_url,
                token,
                "/employee/employment",
                {
                    "employee": {"id": emp_id},
                    "startDate": payload["startDate"],
                },
            )
            print(f"employment -> {r_emp.status_code}: {r_emp.text[:200]}")


def do_create_project(base_url: str, token: str, payload: dict) -> None:
    body = {
        "name": payload.get("name", "New Project"),
        "startDate": payload.get("startDate", "2026-03-20"),
        "endDate": payload.get("endDate", "2027-03-20"),
    }
    body["number"] = payload.get("number", "")
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
    if payload.get("customerName"):
        r_cust = tx_get(base_url, token, "/customer",
                       {"name": payload["customerName"], "count": 1})
        if r_cust.status_code == 200:
            customers = r_cust.json().get("values", [])
            if customers:
                body["customer"] = {"id": customers[0]["id"]}
    r = tx_post(base_url, token, "/project", body)
    if r.status_code == 422 and "i bruk" in r.text:
        print("Project name/number in use, retrying without number")
        body.pop("number", None)
        r = tx_post(base_url, token, "/project", body)


def do_create_department(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "New Department")}
    if payload.get("departmentNumber"):
        body["departmentNumber"] = payload["departmentNumber"]
    tx_post(base_url, token, "/department", body)


def do_create_invoice(base_url: str, token: str, payload: dict) -> None:
    customer_name = payload.get("customer_name", "Unknown Customer")
    invoice_date = payload.get("invoiceDate", "2025-03-20")
    due_date = payload.get("invoiceDueDate", "2025-04-20")

    # Step 1: find or create customer — one GET /customer max; same customer_id for order + invoice
    r = tx_get(base_url, token, "/customer", {"name": customer_name, "count": 1})
    customer_id = None
    if r.status_code == 200:
        customers = r.json().get("values", [])
        if customers:
            customer_id = customers[0]["id"]
    if customer_id is None:
        r2 = tx_post(base_url, token, "/customer",
                     {"name": customer_name, "isCustomer": True})
        if r2.status_code in (200, 201):
            customer_id = r2.json().get("value", {}).get("id")
    if customer_id is None:
        print("Could not find or create customer for invoice - aborting")
        return

    raw_orders = payload.get("orders") or []
    if not raw_orders:
        raw_orders = [{"description": "Service",
                       "unitPriceExcludingVatCurrency": 0, "count": 1}]

    # Step 2: create order
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
        print(
            f"Order creation failed ({r3.status_code}), retrying with key 'lines' "
            f"instead of 'orderLines'"
        )
        order_body = {
            "customer": {"id": customer_id},
            "orderDate": invoice_date,
            "deliveryDate": invoice_date,
            "lines": order_lines,
        }
        r3 = tx_post(base_url, token, "/order", order_body)
    if r3.status_code not in (200, 201):
        print(f"Order creation failed ({r3.status_code})")
        return
    order_id = r3.json().get("value", {}).get("id")
    if not order_id:
        print("No order ID returned")
        return
    print(f"Order created successfully, order_id={order_id} — proceeding to invoice")

    # Step 3: invoice from order
    invoice_body = {
        "invoiceDate": invoice_date,
        "invoiceDueDate": due_date,
        "customer": {"id": customer_id},
        "orders": [{"id": order_id}],
    }
    r_inv = tx_post(base_url, token, "/invoice", invoice_body)
    print(f"Invoice attempt -> {r_inv.status_code}: {r_inv.text[:200]}")
    if r_inv.status_code in (200, 201):
        print("Invoice created successfully!")
    elif r_inv.status_code == 422 and "bankkontonummer" in r_inv.text:
        print("Bank account missing - falling back to ledger voucher")

        VAT_ACCOUNT_MAP = {
            25: 3000,
            15: 3100,
            0: 3000,  # exempt: 3000 with no vatType
        }

        description = f"Faktura {customer_name} {invoice_date} (ordre {order_id})"
        postings = []
        acct_1500 = get_account_id(base_url, token, 1500)
        if acct_1500 is None:
            acct_1500 = 1500
        vat_25 = get_vat_type_id_by_number(base_url, token, "3")
        vat_15 = get_vat_type_id_by_number(base_url, token, "33")

        row = 1
        for item in raw_orders:
            amount = item.get("unitPriceExcludingVatCurrency", 0) * item.get(
                "count", 1
            )
            raw_vat = item.get("vatRate", 25)
            if raw_vat is None:
                vat_rate = 25
            else:
                vat_rate = int(round(float(raw_vat)))
            if vat_rate not in (0, 15, 25):
                vat_rate = 25

            revenue_account_num = VAT_ACCOUNT_MAP.get(vat_rate, 3000)
            acct_revenue = get_account_id(base_url, token, revenue_account_num)

            if acct_revenue is None and revenue_account_num != 3000:
                print(
                    f"Account {revenue_account_num} not found, falling back to 3000"
                )
                acct_revenue = get_account_id(base_url, token, 3000)
            if acct_revenue is None:
                acct_revenue = 3000

            debit = {
                "row": row,
                "date": invoice_date,
                "description": description,
                "account": {"id": acct_1500},
                "amount": amount,
                "amountCurrency": amount,
                "customer": {"id": customer_id},
            }
            postings.append(debit)
            row += 1

            credit = {
                "row": row,
                "date": invoice_date,
                "description": description,
                "account": {"id": acct_revenue},
                "amount": -amount,
                "amountCurrency": -amount,
            }
            if vat_rate == 25 and vat_25:
                credit["vatType"] = {"id": vat_25}
            elif vat_rate == 15 and vat_15:
                credit["vatType"] = {"id": vat_15}

            postings.append(credit)
            row += 1

        voucher_body = {
            "date": invoice_date,
            "description": description,
            "postings": postings,
        }
        try:
            r_voucher = tx_post(base_url, token, "/ledger/voucher", voucher_body)
            print(
                f"Invoice fallback voucher -> {r_voucher.status_code}: "
                f"{r_voucher.text[:300]}"
            )
        except Exception as e:
            print(f"Invoice fallback voucher EXCEPTION: {e}")
    else:
        print("Invoice failed")


def do_create_ledger_posting(base_url: str, token: str, payload: dict) -> None:
    date = payload.get("date", "2025-03-20")
    description = payload.get("description", "Manual posting")
    amount = payload.get("amount", 0)
    debit = payload.get("debitAccount", "1500")
    credit = payload.get("creditAccount", "4000")

    postings = [
        make_posting(base_url, token, date, description, int(debit), amount, row=1),
        make_posting(base_url, token, date, description, int(credit), -amount, row=2),
    ]
    body = {
        "date": date,
        "description": description,
        "postings": postings,
    }
    if payload.get("departmentName"):
        r_dept = tx_get(
            base_url,
            token,
            "/department",
            {"name": payload["departmentName"], "count": 1},
        )
        if r_dept.status_code == 200:
            depts = r_dept.json().get("values", [])
            if depts:
                dept_ref = {"id": depts[0]["id"]}
                for posting in body["postings"]:
                    posting["department"] = dept_ref
                print(f"Added department {depts[0]['id']} to all postings")
    elif payload.get("departmentId") is not None:
        try:
            dept_ref = {"id": int(payload["departmentId"])}
        except (TypeError, ValueError):
            dept_ref = None
        if dept_ref:
            for posting in body["postings"]:
                posting["department"] = dept_ref
    r = tx_post(base_url, token, "/ledger/voucher", body)

    # If employee sub-ledger required, add employee to expense postings
    if r.status_code == 422 and "employee" in r.text.lower() and "mangler" in r.text:
        print("Employee sub-ledger required, looking up first employee")
        r_emp = tx_get(base_url, token, "/employee", {"count": 1})
        if r_emp.status_code == 200:
            emps = r_emp.json().get("values", [])
            if emps:
                emp_ref = {"id": emps[0]["id"]}
                for posting in body["postings"]:
                    posting["employee"] = emp_ref
                r = tx_post(base_url, token, "/ledger/voucher", body)
                print(f"Retry with employee -> {r.status_code}: {r.text[:200]}")


def do_create_accounting_dimension(base_url: str, token: str, payload: dict) -> None:
    """Create a department/dimension and its values."""
    name = payload.get("name", "Dimension")
    values = payload.get("values", [])

    r = tx_post(base_url, token, "/department", {"name": name})
    print(f"Create dimension {name} -> {r.status_code}")

    for value in values:
        r2 = tx_post(base_url, token, "/department", {"name": value})
        print(f"Create dimension value {value} -> {r2.status_code}")


def do_create_payroll(base_url: str, token: str, payload: dict) -> None:
    employee_email = payload.get("employeeEmail", "")
    base_salary = payload.get("baseSalary", 0)
    bonus = payload.get("bonus", 0)
    date = payload.get("date", "2025-03-20")

    # Find employee
    employee_id = None
    employee_name = "Employee"
    if employee_email:
        r = tx_get(base_url, token, "/employee", {"email": employee_email})
        if r.status_code == 200:
            employees = r.json().get("values", [])
            if employees:
                employee_id = employees[0]["id"]
                employee_name = employees[0].get("displayName", "Employee")

    if not employee_id:
        print("No employee found for payroll")
        return

    dt = datetime.strptime(date, "%Y-%m-%d")

    r2 = tx_post(base_url, token, "/salary/transaction", {
        "year": dt.year,
        "month": dt.month,
        "payslips": [{"employee": {"id": employee_id}, "amount": base_salary}],
    })
    print(f"salary/transaction -> {r2.status_code}: {r2.text[:200]}")
    if r2.status_code in (200, 201):
        return

    # Fallback: manual ledger voucher on salary accounts
    description = f"Lønn {employee_name} {date[:7]}"
    postings = [
        make_posting(base_url, token, date, description, 5000, base_salary, row=1),
        make_posting(base_url, token, date, description, 2910, -base_salary, row=2),
    ]
    if bonus:
        bonus_desc = f"Bonus {employee_name} {date[:7]}"
        postings.append(make_posting(base_url, token, date, bonus_desc, 5000, bonus, row=3))
        postings.append(make_posting(base_url, token, date, bonus_desc, 2910, -bonus, row=4))

    for posting in postings:
        if posting["account"].get("id") and employee_id:
            # Check if this is the 2910 posting (credit side)
            if posting["amount"] < 0:
                posting["employee"] = {"id": employee_id}

    voucher_body = {
        "date": date,
        "description": description,
        "postings": postings
    }
    r3 = tx_post(base_url, token, "/ledger/voucher", voucher_body)
    print(f"ledger/voucher payroll fallback -> {r3.status_code}: {r3.text[:200]}")


def do_create_travel_expense(base_url: str, token: str, payload: dict) -> None:
    employee_email = payload.get("employeeEmail", "")
    date = payload.get("date", "2025-03-20")
    title = payload.get("description", "Travel expense")

    # Find employee by email or use first
    employee_id = None
    if employee_email:
        r = tx_get(base_url, token, "/employee", params={"email": employee_email})
        if r.status_code == 200:
            employees = r.json().get("values", [])
            if employees:
                employee_id = employees[0]["id"]
    if not employee_id:
        r = tx_get(base_url, token, "/employee", {"count": 1})
        if r.status_code == 200:
            employees = r.json().get("values", [])
            if employees:
                employee_id = employees[0]["id"]
    if not employee_id:
        print("No employee found for travel expense")
        return

    body = {
        "employee": {"id": employee_id},
        "title": title,
        "date": date,
        "isCompleted": False,
    }
    r2 = tx_post(base_url, token, "/travelExpense", body)
    print(f"travelExpense -> {r2.status_code}: {r2.text[:200]}")


def do_register_payment(base_url: str, token: str, payload: dict) -> None:
    amount = payload.get("amount", 0)
    date = payload.get("date", "2025-03-20")
    customer_name = payload.get("customer_name", "")

    # Find invoice
    invoice_id = payload.get("invoiceId")
    invoice_amount = amount
    customer_id = None

    if not invoice_id and customer_name:
        r = tx_get(base_url, token, "/customer", {"name": customer_name, "count": 1})
        if r.status_code == 200:
            customers = r.json().get("values", [])
            if customers:
                customer_id = customers[0]["id"]
                r2 = tx_get(base_url, token, "/invoice", {
                    "customerId": customer_id,
                    "invoiceDateFrom": "2020-01-01",
                    "invoiceDateTo": "2030-12-31",
                    "count": 1,
                })
                if r2.status_code == 200:
                    invoices = r2.json().get("values", [])
                    if invoices:
                        invoice_id = invoices[0]["id"]
                        invoice_amount = invoices[0].get("amount", amount) or amount

    if not invoice_id:
        r3 = tx_get(base_url, token, "/invoice", {
            "invoiceDateFrom": "2020-01-01",
            "invoiceDateTo": "2030-12-31",
            "count": 1
        })
        if r3.status_code == 200:
            invoices = r3.json().get("values", [])
            if invoices:
                invoice_id = invoices[0]["id"]
                invoice_amount = invoices[0].get("amount", amount) or amount

    if not invoice_id:
        print("No invoice found for payment")
        return

    use_amount = invoice_amount if invoice_amount else amount
    description = f"Betaling {customer_name} {date}"

    if use_amount < 0 or amount < 0:
        r_vouchers = tx_get(
            base_url,
            token,
            "/ledger/voucher",
            {
                "dateFrom": "2020-01-01",
                "dateTo": "2030-12-31",
                "count": 10,
            },
        )
        if r_vouchers.status_code == 200:
            vouchers = r_vouchers.json().get("values", [])
            for voucher in vouchers:
                vid = voucher.get("id")
                if vid:
                    rev_url = f"{base_url.rstrip('/')}/ledger/voucher/{vid}/:reverse"
                    r_rev = requests.put(
                        rev_url,
                        auth=tx_auth(token),
                        params={"date": date},
                        timeout=30,
                    )
                    print(
                        f"Reverse voucher {vid} -> {r_rev.status_code}: "
                        f"{r_rev.text[:200]}"
                    )
                    if r_rev.status_code in (200, 201):
                        return
        use_amount = abs(use_amount)
        postings = [
            make_posting(base_url, token, date, description, 1500, use_amount, row=1),
            make_posting(base_url, token, date, description, 1920, -use_amount, row=2),
        ]
        if customer_id:
            postings[0]["customer"] = {"id": customer_id}
        voucher_body = {
            "date": date,
            "description": f"Reversering {description}",
            "postings": postings,
        }
        r_v = tx_post(base_url, token, "/ledger/voucher", voucher_body)
        print(f"Reversal voucher -> {r_v.status_code}: {r_v.text[:200]}")
        return

    # First try /:payment as query params (may work on some proxies)
    url = f"{base_url.rstrip('/')}/invoice/{invoice_id}/:payment"
    r_pay = requests.put(url, auth=tx_auth(token), params={
        "paymentDate": date,
        "paymentTypeId": 1,
        "paidAmount": invoice_amount,
    }, timeout=30)
    print(f"PUT /:payment -> {r_pay.status_code}: {r_pay.text[:200]}")
    if r_pay.status_code in (200, 201):
        return

    # Fallback: ledger voucher — debit bank (1920), credit AR (1500)
    postings = [
        make_posting(base_url, token, date, description, 1920, use_amount, row=1),
        make_posting(base_url, token, date, description, 1500, -use_amount, row=2),
    ]
    # Account 1500 is customer sub-ledger — requires customer reference
    if customer_id:
        postings[1]["customer"] = {"id": customer_id}
    voucher_body = {
        "date": date,
        "description": description,
        "postings": postings,
    }
    r_v = tx_post(base_url, token, "/ledger/voucher", voucher_body)
    print(f"Payment voucher -> {r_v.status_code}: {r_v.text[:200]}")


def do_reverse_payment(base_url: str, token: str, payload: dict) -> None:
    date = payload.get("date", "2026-03-20")

    r_v = tx_get(
        base_url,
        token,
        "/ledger/voucher",
        {"dateFrom": "2020-01-01", "dateTo": "2030-12-31", "count": 10},
    )
    if r_v.status_code == 200:
        vouchers = r_v.json().get("values", [])
        payment_vouchers = [
            v for v in vouchers if "etaling" in v.get("description", "")
        ]
        all_vouchers = payment_vouchers if payment_vouchers else vouchers
        for voucher in all_vouchers:
            vid = voucher.get("id")
            if vid:
                rev_url = f"{base_url.rstrip('/')}/ledger/voucher/{vid}/:reverse"
                r_rev = requests.put(
                    rev_url,
                    auth=tx_auth(token),
                    params={"date": date},
                    timeout=30,
                )
                print(
                    f"Reverse voucher {vid} -> {r_rev.status_code}: {r_rev.text[:200]}"
                )
                if r_rev.status_code in (200, 201):
                    return
    print("Could not reverse any voucher")


def do_create_credit_note(base_url: str, token: str, payload: dict) -> None:
    date = payload.get("date", "2025-03-20")
    customer_name = payload.get("customerName", "")
    invoice_id = payload.get("invoiceId")
    invoice_amount = payload.get("amount", 0)
    customer_id = None

    if not invoice_id and customer_name:
        r = tx_get(base_url, token, "/customer", {"name": customer_name, "count": 1})
        if r.status_code == 200:
            customers = r.json().get("values", [])
            if customers:
                customer_id = customers[0]["id"]
                r2 = tx_get(base_url, token, "/invoice", {
                    "customerId": customer_id,
                    "invoiceDateFrom": "2020-01-01",
                    "invoiceDateTo": "2030-12-31",
                    "count": 1
                })
                if r2.status_code == 200:
                    invoices = r2.json().get("values", [])
                    if invoices:
                        invoice_id = invoices[0]["id"]

    if not invoice_id:
        r3 = tx_get(base_url, token, "/invoice", {
            "invoiceDateFrom": "2020-01-01",
            "invoiceDateTo": "2030-12-31",
            "count": 1
        })
        if r3.status_code == 200:
            invoices = r3.json().get("values", [])
            if invoices:
                invoice_id = invoices[0]["id"]

    if invoice_id and not invoice_amount:
        r_inv = tx_get(base_url, token, f"/invoice/{invoice_id}")
        if r_inv.status_code == 200:
            inv_data = r_inv.json().get("value", {})
            invoice_amount = (
                inv_data.get("amount", 0)
                or inv_data.get("amountCurrency", 0)
                or inv_data.get("amountExcludingVat", 0)
            )
            print(f"Invoice amount from API: {invoice_amount}")

    if not invoice_id:
        print("No invoice found - creating credit voucher directly")
        customer_name_for_voucher = customer_name or "Unknown"
        pd = f"Kreditnota {customer_name_for_voucher}"
        vat_type_id = lookup_vat_type_mva3(base_url, token)
        postings = [
            make_posting(
                base_url, token, date, pd, 3000, invoice_amount, row=1, vat_type_id=vat_type_id
            ),
            make_posting(base_url, token, date, pd, 1500, -invoice_amount, row=2),
        ]
        cn_customer_id = customer_id
        if not cn_customer_id and customer_name:
            r_cust = tx_get(base_url, token, "/customer", {"name": customer_name, "count": 1})
            if r_cust.status_code == 200:
                custs = r_cust.json().get("values", [])
                if custs:
                    cn_customer_id = custs[0]["id"]
        if cn_customer_id:
            postings[1]["customer"] = {"id": cn_customer_id}
        voucher_body = {
            "date": date,
            "description": f"Kreditnota {customer_name_for_voucher} {date}",
            "postings": postings,
        }
        tx_post(base_url, token, "/ledger/voucher", voucher_body)
        return

    url = f"{base_url.rstrip('/')}/invoice/{invoice_id}/:createCreditNote"
    r4 = requests.put(url, auth=tx_auth(token), params={"date": date}, timeout=30)
    print(f"PUT /:createCreditNote -> {r4.status_code}: {r4.text[:200]}")
    if r4.status_code in (200, 201):
        return

    # /:createCreditNote blocked or failed — fall back to reversal voucher
    cn_amount = invoice_amount if invoice_amount else 10000
    description = f"Kreditnota {customer_name} {date}"
    vat_type_id = lookup_vat_type_mva3(base_url, token)
    postings = [
        make_posting(
            base_url, token, date, description, 3000, cn_amount, row=1, vat_type_id=vat_type_id
        ),
        make_posting(base_url, token, date, description, 1500, -cn_amount, row=2),
    ]
    cn_customer_id = customer_id
    if not cn_customer_id and customer_name:
        r_cust = tx_get(base_url, token, "/customer", {"name": customer_name, "count": 1})
        if r_cust.status_code == 200:
            custs = r_cust.json().get("values", [])
            if custs:
                cn_customer_id = custs[0]["id"]
    if cn_customer_id:
        postings[1]["customer"] = {"id": cn_customer_id}
    voucher_body = {
        "date": date,
        "description": description,
        "postings": postings,
    }
    r_v = tx_post(base_url, token, "/ledger/voucher", voucher_body)
    print(f"Credit note voucher -> {r_v.status_code}: {r_v.text[:200]}")


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


def do_log_hours(base_url: str, token: str, payload: dict) -> None:
    hours = payload.get("hours", 0)
    date = payload.get("date", "2025-03-20")
    employee_email = payload.get("employeeEmail", "")
    project_name = payload.get("projectName", "")

    # Find employee
    employee_id = None
    if employee_email:
        r = tx_get(base_url, token, "/employee", {"email": employee_email})
        if r.status_code == 200:
            employees = r.json().get("values", [])
            if employees:
                employee_id = employees[0]["id"]

    # Find project
    project_id = None
    if project_name:
        r = tx_get(base_url, token, "/project", {"name": project_name, "count": 1})
        if r.status_code == 200:
            projects = r.json().get("values", [])
            if projects:
                project_id = projects[0]["id"]

    if project_id:
        r_proj = tx_get(base_url, token, f"/project/{project_id}")
        if r_proj.status_code == 200:
            proj = r_proj.json().get("value", {})
            proj_start = proj.get("startDate") or date
            proj_end = proj.get("endDate") or "2099-12-31"
            # Use project start date if our date is before it
            if date < proj_start:
                date = proj_start
                print(f"Adjusted timesheet date to project start: {date}")
            # Use project end date if our date is after it
            elif date > proj_end:
                date = proj_end
                print(f"Adjusted timesheet date to project end: {date}")

    # Find or use default activity
    activity_id = None
    activity_name = payload.get("activityName", "")
    if activity_name:
        r = tx_get(base_url, token, "/activity", {"name": activity_name, "count": 1})
        if r.status_code == 200:
            activities = r.json().get("values", [])
            if activities:
                activity_id = activities[0]["id"]
    if not activity_id:
        r = tx_get(base_url, token, "/activity", {"count": 1})
        if r.status_code == 200:
            activities = r.json().get("values", [])
            if activities:
                activity_id = activities[0]["id"]

    if not employee_id or not project_id or not activity_id:
        print(f"Missing required IDs for timekeeping: employee={employee_id} project={project_id} activity={activity_id}")
        return

    body = {
        "date": date,
        "hours": hours,
        "employee": {"id": employee_id},
        "project": {"id": project_id},
        "activity": {"id": activity_id},
    }
    if payload.get("hourlyRate"):
        body["hourlyRate"] = payload["hourlyRate"]

    tx_post(base_url, token, "/timesheet/entry", body)


def do_create_contact(base_url: str, token: str, payload: dict) -> None:
    body = {
        "firstName": payload.get("firstName", "Unknown"),
        "lastName": payload.get("lastName", "Contact"),
    }
    for field in ("email", "phoneNumber"):
        if payload.get(field):
            body[field] = payload[field]
    if payload.get("customerName"):
        r = tx_get(base_url, token, "/customer",
                   {"name": payload["customerName"], "count": 1})
        if r.status_code == 200:
            customers = r.json().get("values", [])
            if customers:
                body["customer"] = {"id": customers[0]["id"]}
    elif payload.get("customerId"):
        body["customer"] = {"id": payload["customerId"]}
    tx_post(base_url, token, "/contact", body)


def do_delete_customer(base_url: str, token: str, payload: dict) -> None:
    name = payload.get("name", "")
    r = tx_get(base_url, token, "/customer", {"name": name, "count": 1})
    if r.status_code != 200:
        return
    customers = r.json().get("values", [])
    if not customers:
        print(f"No customer found: {name}")
        return
    customer_id = customers[0]["id"]
    url = f"{base_url.rstrip('/')}/customer/{customer_id}"
    r2 = requests.delete(url, auth=tx_auth(token), timeout=30)
    print(f"DELETE /customer/{customer_id} -> {r2.status_code}")


def do_delete_employee(base_url: str, token: str, payload: dict) -> None:
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
    url = f"{base_url.rstrip('/')}/employee/{employee_id}"
    r2 = requests.delete(url, auth=tx_auth(token), timeout=30)
    print(f"DELETE /employee/{employee_id} -> {r2.status_code}")


def do_create_product_with_price(base_url: str, token: str, payload: dict) -> None:
    body = {"name": payload.get("name", "Unknown Product")}
    if payload.get("priceExcludingVatCurrency") is not None:
        body["priceExcludingVatCurrency"] = payload["priceExcludingVatCurrency"]
    if payload.get("costExcludingVatCurrency") is not None:
        body["costExcludingVatCurrency"] = payload["costExcludingVatCurrency"]
    if payload.get("number"):
        body["number"] = payload["number"]
    tx_post(base_url, token, "/product", body)


def do_register_supplier_invoice(base_url: str, token: str, payload: dict) -> None:
    supplier_name = payload.get("supplierName", "Unknown Supplier")
    date = payload.get("date", "2025-03-20")
    amount_with_vat = payload.get("amount", 0)
    vat_percent = payload.get("vatPercent", 25)
    account_code = payload.get("accountCode", "6340")
    invoice_number = payload.get("invoiceNumber", "")

    # Calculate amounts
    vat_factor = vat_percent / (100 + vat_percent)
    vat_amount = round(amount_with_vat * vat_factor, 2)
    net_amount = round(amount_with_vat - vat_amount, 2)

    # Find or create supplier
    r = tx_get(base_url, token, "/customer",
               {"name": supplier_name, "count": 1})
    supplier_id = None
    if r.status_code == 200:
        suppliers = r.json().get("values", [])
        if suppliers:
            supplier_id = suppliers[0]["id"]
    if not supplier_id:
        org_no = payload.get("organizationNumber", "")
        body = {"name": supplier_name, "isSupplier": True}
        if org_no:
            body["organizationNumber"] = org_no
        r_supplier = tx_post(base_url, token, "/customer", body)
        if r_supplier.status_code in (200, 201):
            val = r_supplier.json().get("value", {})
            sid = val.get("id")
            if sid:
                supplier_id = sid
                put_body = {
                    k: v
                    for k, v in val.items()
                    if k not in ("url", "changes", "displayName")
                }
                put_body["isCustomer"] = False
                put_url = f"{base_url.rstrip('/')}/customer/{sid}"
                r_put = requests.put(
                    put_url, auth=tx_auth(token), json=put_body, timeout=30
                )
                print(f"PUT supplier isCustomer=False -> {r_put.status_code}")

    # Create voucher with postings
    description = f"{invoice_number} - {supplier_name}" if invoice_number else supplier_name
    postings = [
        make_posting(base_url, token, date, description, int(account_code), net_amount, row=1),
        make_posting(
            base_url,
            token,
            date,
            f"VAT {vat_percent}% - {description}",
            2700,
            vat_amount,
            row=2,
        ),
        make_posting(base_url, token, date, description, 2400, -amount_with_vat, row=3),
    ]
    # Account 2400 is supplier sub-ledger — requires supplier reference
    if supplier_id:
        postings[2]["supplier"] = {"id": supplier_id}
    voucher_body = {
        "date": date,
        "description": description,
        "postings": postings,
    }
    tx_post(base_url, token, "/ledger/voucher", voucher_body)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

ACTION_MAP = {
    "create_customer": do_create_customer,
    "create_supplier": do_create_supplier,
    "create_product": do_create_product,
    "update_product": do_update_product,
    "create_employee": do_create_employee,
    "create_project": do_create_project,
    "create_department": do_create_department,
    "create_accounting_dimension": do_create_accounting_dimension,
    "create_payroll": do_create_payroll,
    "create_invoice": do_create_invoice,
    "create_ledger_posting": do_create_ledger_posting,
    "create_travel_expense": do_create_travel_expense,
    "register_payment": do_register_payment,
    "reverse_payment": do_reverse_payment,
    "create_credit_note": do_create_credit_note,
    "update_customer": do_update_customer,
    "delete_travel_expense": do_delete_travel_expense,
    "update_employee": do_update_employee,
    "log_hours": do_log_hours,
    "create_contact": do_create_contact,
    "delete_customer": do_delete_customer,
    "delete_employee": do_delete_employee,
    "create_product_with_price": do_create_product_with_price,
    "register_supplier_invoice": do_register_supplier_invoice,
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
    base_url = body.tripletex_credentials.base_url
    token = body.tripletex_credentials.session_token
    print(f"=== incoming prompt: {body.prompt!r} ===")
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
    print(f"=== Solve complete: {len(plan.get('actions', []))} actions dispatched ===")
    return {"status": "completed"}
