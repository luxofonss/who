
# QA Roles and Responsibilities

## Main Responsibilities
Analyze requirements and create test cases
Perform manual and automated testing
Report bugs and follow up on fixes
Ensure full test coverage
Update test documentation
Support the team in ensuring quality

## Best Practices for Writing Test Cases

## 1. Standard Test Case Structure
**Test Case ID**: Unique identifier
**Test Case Name**: Clear, descriptive name
**Preconditions**: Required setup or states before test
**Test Steps**: Detailed execution steps
**Test Data**: Input data needed for testing
**Expected Results**: Anticipated outcome
**Actual Results**: Actual outcome during testing
**Status**: Pass / Fail / Blocked

## 2. Principles for Test Case Design
Ensure completeness
Cover normal (happy path) scenarios
Cover edge cases
Test error scenarios
Include boundary value conditions
Keep test cases independent
Ensure readability and maintainability

## 3. Types of Test Cases
**Positive Test Cases**: Validate correct behavior
**Negative Test Cases**: Validate error handling
**Boundary Test Cases**: Test at value limits
**Performance Test Cases**: Measure performance
**Security Test Cases**: Assess security

## 4. Quality Criteria for Test Cases
Clear and easy to understand
Executable
Repeatable
Traceable
Maintainable

# 🧪 Special Notes for API Test Case Design

## 1. HTTP Methods
`GET`: Retrieve data
`POST`: Create data
`PUT`: Update data
`DELETE`: Delete data
`PATCH`: Partial update

## 2. Status Codes
`200`: OK
`201`: Created
`400`: Bad Request
`401`: Unauthorized
`403`: Forbidden
`404`: Not Found
`500`: Internal Server Error

## 3. API Test Scenarios
Valid input with valid response
Invalid input with error response
Missing required fields
Invalid data types
Boundary values
Load and performance testing
Security vulnerability testing

## Checklist for Test Case Coverage

[ ] All core functionalities are tested
[ ] All error cases are covered
[ ] Boundary conditions tested
[ ] Edge cases tested
[ ] Security scenarios tested
[ ] Basic performance tested
[ ] Compatibility tested
[ ] Usability tested
[ ] Test cases are executable
[ ] Test cases are repeatable
[ ] Test cases are traceable
[ ] Test cases are maintainable

# 🧰 Test Case Writing Techniques

## Requirement Analysis

### How to Analyze Requirement Documents
**FSD**: Functional Specification Document
**SRS**: Software Requirements Specification
**User Stories** (Agile)
**Acceptance Criteria**

**Tips:**
Read carefully, line by line
Highlight key terms
Identify "if-then" logic
Clarify ambiguous wording
Find special cases

### Techniques to Clarify Requirements
“What happens if...?”
“Are there any limits for...?”
“Are there exceptions?”
“How do we identify...?”
“Are there performance/security requirements?”

### Constraints and Assumptions
**Constraints**: Rules that must be followed
**Assumptions**: Presumed conditions (e.g., user is logged in)

# 🧪 Black Box Test Techniques

## 1. Equivalence Partitioning
Divide inputs into equivalent groups
Test one representative per group

## 2. Boundary Value Analysis
Test values at, above, and below boundaries

## 3. Decision Table Testing
List all conditions and outcomes
Create test cases for each combination

## 4. State Transition Testing
Test transitions between system states

## 5. Use Case Testing
Write tests based on use cases (main, alternate, and exception flows)

# 🔍 White Box Testing (Basic Overview)

## 1. Statement Coverage
Ensure every line of code runs at least once

## 2. Branch Coverage
Ensure each `if/else` path is executed

## 3. Path Coverage
Ensure every possible execution path is tested

**Why Testers Should Understand White Box Testing:**
Helps design better test cases
Know which parts need more testing
Collaborate with developers on coverage

# 🕵️‍♀️ Exploratory Testing

**When**: When requirements are unclear or time is limited
**How**:
  Start with a basic checklist
  Document tested items
  Focus on core features
  Try unexpected inputs

# 🧠 Experience-Based Testing

## 1. Error Guessing
Predict common issues based on experience
Common issues:
  Null pointer
  Division by zero
  Buffer overflow
  SQL Injection
  XSS
  Race conditions
  Memory leaks

## 2. Checklist-Based Testing
Helps avoid missing test cases
Build checklist from:
  Experience
  Best practices
  Functional areas

## 3. Risk-Based Testing
Prioritize tests based on:
  User impact
  Business impact
  Usage frequency
  Code complexity
  History of defects

# 📘 Real-World Examples

### Login Function
Equivalence: Valid/Invalid/Empty usernames
Boundary: Min/Max username lengths
Decision Table: Username/password combinations
Error Guessing: SQL Injection, XSS

### Search Function
Boundary: Empty, very long, special characters
State Transition: Loading → Success/Error
Exploratory: Try unusual or Unicode keywords

# 🧾 Effective Test Case Structure

## Required Fields

### 1. Test Case ID
Unique identifier (e.g., `TC_001`)

### 2. Title
Short, descriptive purpose

### 3. Description
Purpose, scope, conditions

### 4. Preconditions
Required data/system state

### 5. Test Steps
Clear, numbered steps
Include input data

### 6. Expected Result
Must be SMART:
  Specific
  Measurable
  Achievable
  Relevant
  Time-bound

### 7. Actual Result
Observed result

### 8. Status
Pass / Fail / Blocked / Not Executed

# 🧠 SMART Rule for Expected Result

1. **Specific**: Clear, no ambiguity
2. **Measurable**: Verifiable outcome
3. **Achievable**: Realistic
4. **Relevant**: Aligned with business goals
5. **Time-bound**: Response time is specified

# ✍️ Writing Clear Test Cases

Use simple language
One action per step
Command form: "Enter", "Click", "Verify"
Logical structure and bullet points
Include test data and environment

# ❌ Writing Negative Test Cases

## Purpose:
Ensure proper error handling
Prevent system crashes
Verify meaningful error messages

## Types:
Invalid input
Empty input
Too long input
Wrong format
Non-existent data

## Examples:
Login with non-existent username
Register with invalid email
Search with empty keyword
Upload oversized file
Enter negative age

# 🧪 Writing Edge Cases

## Definition:
Values at the edge or limit
Detect logic issues at transition points

## Examples:
Min length username (1 char)
Max length username (50 chars)
Password with only spaces
Email with only special characters
Age = 0 or 150

# 🧾 Sample Test Case Template

```
Test Case ID: TC_001
Title: Verify login success with valid credentials
Description: Verify that user can login successfully with valid credentials
Pre-conditions:
User account is registered
System is up and running

Test Steps:
1. Open login page
2. Enter username: "testuser"
3. Enter password: "password123"
4. Click "Login" button

Expected Result:
Display message: "Login successful"
Redirect to dashboard page
Show logged-in user info
Return status code: 200

Actual Result: [To be filled during test execution]
Status: Pass/Fail/Blocked
```

## Best Practices Summary

## General Principles
One test case per function
Tests are independent and repeatable
Results should be predictable

## Naming Rules
Clear and descriptive
Avoid vague terms
Use business keywords

## Writing Steps
One action per step
Logical order
Include test data

## Expected Result
Follow SMART
Include system behavior and data output

## Critical Thinking
Always question assumptions
Consider various perspectives

## Asking the Right Questions
Clarify requirements
Understand business logic

## Improving Critical Thinking
Analyze carefully
Discuss with team/stakeholders
Study case studies
Learn from experienced testers
