# REST API Document

# API Make payment

## 0\. Logic flow
detail logic flow of code in the API

## 1\. API Specs

| Name | Make payment |
| :---- | :---- |
| Description | Create a transaction between 2 normal accounts |
| URL | https://{{ronin\_engineer}}/payment-service/v1/payments |
| Method | POST |
| Header | Content-Type: application/json Idempotency-Key: string (example: oc8tKg1P2FV44hpj) |
| Params | page: int (default \= 0\) size: int (default \= 10\)  |
| Request Body | {     "debit\_account": "acc01",     "credit\_account": "acc02",     "amount": 1000 } |
| Response | Status Code: 200 {     "meta": {         "code": "200000",         "type": "SUCESS",         "message": "Sucess",         "service\_id": "payment-service",         "extra\_meta": {}     },     "data": {         "transaction\_id": "FT2308163234",         "debit\_account": "acc01",         "credit\_account": "acc02",         "amount": 1000     } } Status Code: 401 {     "meta": {         "code": "400001",         "type": "INSUFFICIENT\_DEBIT\_AMOUNT",         "message": "Debit account has an insufficient amount of balance",         "service\_id": "payment-service",         "extra\_meta": {}     },     "data": null } |
| Sample | curl \--location \--request GET 'https://roninengineer.dev/products/jobs/export?name=pin' \\ \--header 'Content-Type: application/json' \\ \--data '{     "job\_id": "j001",     "status": "PROCESSING",     "issued\_at": 1692163008000 }' |

## 2\. Request Body

| Parameter | Type | Required | Default | Description |
| :---- | :---- | :---- | :---- | :---- |
| debit\_account | String | x |  | Tài khoản trừ tiền. Business logic. Format. Sample |
| credit\_account | String | x |  | Tài khoản cộng tiền |
| amount | Int | x | 0 | Số tiền |

## 3\. Response Body

| Parameter | Type | Required | Default | Description |
| :---- | :---- | :---- | :---- | :---- |
| transaction\_id | String | x |  | Mã giao dịch |
| debit\_account | String | x |  | Tài khoản ghi nợ |
| credit\_account | String | x |  | Tài khoản ghi có |
| amount | Int | x | 0 | Số tiền |

## 4\. Errors

| Status Code | Code | Type | Description |
| :---- | :---- | :---- | :---- |
| 200 | 200000 | SUCCESS | Thành công |
| 400 | 400001 | INSUFFICIENT\_DEBIT\_AMOUNT | Tài khoản ghi nợ không đủ tiền |
|  | 401000 | INSUFFICIENT\_CREDIT\_AMOUNT | Tài khoản ghi có bị khoá |

# 