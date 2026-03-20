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


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


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
