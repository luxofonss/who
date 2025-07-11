"""
Test case generation module
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
import google.generativeai as genai
import os
from src.config import Config


@dataclass
class TestCase:
    """Data class for test case information"""
    test_case_key: str
    test_case_name: str
    prepare_file_name: str
    http_method: str
    expected_http_code: str
    request_param_name: str
    request_param_value: str
    request_body: str
    expect_body: str


class TestGenerator:
    """Class for generating test cases using AI"""
    
    def __init__(self, config: Config):
        """
        Initialize test generator
        
        Args:
            config: Configuration object containing AI settings
        """
        self.config = config
        self.provider = getattr(config, 'AI_PROVIDER', os.getenv('AI_PROVIDER', 'openai'))
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        elif self.provider == "gemini":
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(config.GEMINI_MODEL)
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider}")
        self.test_case_counter = 1
    
    def generate_test_cases(self, requirement_analysis: Any, code_analysis: Any, 
                          reference_test_cases: List[str] = None) -> List[TestCase]:
        """
        Generate test cases based on requirement and code analysis
        
        Args:
            requirement_analysis: Analysis results from requirement parser
            code_analysis: Analysis results from code analyzer
            reference_test_cases: List of reference test cases from Jira
            
        Returns:
            List of TestCase objects
        """
        try:
            # Prepare prompt for test case generation
            prompt = self._build_test_generation_prompt(requirement_analysis, code_analysis, reference_test_cases)
            
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert test engineer specializing in API test case generation."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.OPENAI_MAX_TOKENS,
                    temperature=self.config.OPENAI_TEMPERATURE
                )
                test_cases_text = response.choices[0].message.content
            elif self.provider == "gemini":
                response = self.gemini_model.generate_content(prompt)
                test_cases_text = response.text
            else:
                raise ValueError(f"Unsupported AI provider: {self.provider}")
            
            return self._parse_test_cases(test_cases_text, reference_test_cases)
            
        except Exception as e:
            raise ValueError(f"Failed to generate test cases: {str(e)}")
    
    def _build_test_generation_prompt(self, requirement_analysis: Any, code_analysis: Any, 
                                    reference_test_cases: List[str] = None) -> str:
        """
        Build prompt for test case generation
        """
        # Đọc file kiến thức bổ sung
        knowledge_path = "knowledge_testing.txt"
        knowledge_content = ""
        try:
            with open(knowledge_path, "r", encoding="utf-8") as f:
                knowledge_content = f.read()
        except Exception:
            pass  # Nếu không có file thì bỏ qua

        service_name = code_analysis.service_name or requirement_analysis.service_name or "DefaultService"

        prompt = f"""
Hãy sử dụng các kiến thức, quy tắc, checklist sau để sinh test case chuẩn và đầy đủ nhất:
{knowledge_content}

REQUIREMENT ANALYSIS:
- Main business logic: {requirement_analysis.main_business_logic}
- Sub flows: {requirement_analysis.sub_flows}
- Boundary conditions: {requirement_analysis.boundary_conditions}
- Test scenarios: {requirement_analysis.test_scenarios}

CODE ANALYSIS:
- Service name: {service_name}
- Functions changed: {code_analysis.functions_changed}
- API endpoints: {code_analysis.api_endpoints}
- HTTP methods: {code_analysis.http_methods}
- Request parameters: {code_analysis.request_parameters}
- Logic changes: {code_analysis.logic_changes}

"""
        if reference_test_cases:
            prompt += f"""
REFERENCE TEST CASES:
{chr(10).join(f"- {tc}" for tc in reference_test_cases)}

"""
        prompt += f"""
Please generate test cases in the following format:

TC1: [{service_name}] Kiểm tra API login trả status_code = 200 và response ="Login successfull", Trường hợp truyền param hợp lệ username = testuser, password =password123
TC2: [{service_name}] Kiểm tra API login trả status_code = 400 và response ="Invalid credentials", Trường hợp truyền param username = testuser hợp lệ, password =wrongpassword không hợp lệ

For each test case, provide:
1. Test case key (TC1, TC2, etc.)
2. Test case name with service name in brackets
3. Expected HTTP status code
4. Expected response message
5. Test scenario description
6. Request parameters/body

Focus on:
1. Happy path scenarios
2. Error scenarios
3. Boundary conditions
4. Edge cases
5. Different HTTP methods if applicable
6. Various parameter combinations

Generate test cases that are:
- Comprehensive and cover all scenarios
- Non-duplicate with reference test cases
- Clear and descriptive
- Following the specified format
"""
        return prompt
    
    def _parse_test_cases(self, test_cases_text: str, reference_test_cases: List[str] = None) -> List[TestCase]:
        """
        Parse generated test cases text into TestCase objects
        
        Args:
            test_cases_text: Raw test cases text from LLM
            reference_test_cases: Reference test cases to avoid duplication
            
        Returns:
            List of TestCase objects
        """
        test_cases = []
        
        # First, add reference test cases if they exist
        if reference_test_cases:
            for i, ref_tc in enumerate(reference_test_cases, 1):
                test_case = self._convert_reference_to_test_case(ref_tc, i)
                if test_case:
                    test_cases.append(test_case)
                    self.test_case_counter = i + 1
        
        # Parse generated test cases
        lines = test_cases_text.split('\n')
        current_test_case = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a new test case (starts with TC)
            if line.startswith('TC') and ':' in line:
                # Save previous test case if exists
                if current_test_case:
                    test_cases.append(current_test_case)
                
                # Parse new test case
                current_test_case = self._parse_single_test_case(line)
            elif current_test_case:
                # This might be additional information for current test case
                self._enhance_test_case(current_test_case, line)
        
        # Add the last test case
        if current_test_case:
            test_cases.append(current_test_case)
        
        return test_cases
    
    def _parse_single_test_case(self, line: str) -> TestCase:
        """
        Parse a single test case line
        
        Args:
            line: Test case line (e.g., "TC1: [AuthService] Kiểm tra API login...")
            
        Returns:
            TestCase object
        """
        try:
            # Extract test case key
            if ':' in line:
                _, description = line.split(':', 1)
                test_case_key = ""
                test_case_name = description.strip()  # Bỏ tiền tố TCx:
            else:
                test_case_key = ""
                test_case_name = line.strip()
                self.test_case_counter += 1
            
            # Extract service name from brackets
            service_name = "DefaultService"
            if '[' in test_case_name and ']' in test_case_name:
                start = test_case_name.find('[') + 1
                end = test_case_name.find(']')
                if start > 0 and end > start:
                    service_name = test_case_name[start:end]
                    test_case_name = test_case_name[end + 1:].strip()
            
            # Extract HTTP status code
            http_code = "200"
            if "status_code = " in test_case_name:
                import re
                match = re.search(r'status_code\s*=\s*(\d+)', test_case_name)
                if match:
                    http_code = match.group(1)
            
            # Extract HTTP method
            http_method = "POST"
            if any(method in test_case_name.upper() for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]):
                for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    if method in test_case_name.upper():
                        http_method = method
                        break
            
            # Generate prepare file name
            prepare_file_name = f"prepareData_{service_name}_{self.test_case_counter}.csv"
            
            # Extract request body (simplified)
            request_body = "{}"
            if "param" in test_case_name.lower():
                # Try to extract parameters
                import re
                param_match = re.search(r'param\s+(\w+)\s*=\s*(\w+)', test_case_name)
                if param_match:
                    param_name = param_match.group(1)
                    param_value = param_match.group(2)
                    request_body = f'{{"{param_name}": "{param_value}"}}'
            
            # Extract expected response
            expect_body = "{}"
            if "response =" in test_case_name:
                import re
                response_match = re.search(r'response\s*=\s*"([^"]+)"', test_case_name)
                if response_match:
                    response_text = response_match.group(1)
                    expect_body = f'{{"message": "{response_text}"}}'
            
            return TestCase(
                test_case_key=test_case_key,
                test_case_name=test_case_name,
                prepare_file_name=prepare_file_name,
                http_method=http_method,
                expected_http_code=http_code,
                request_param_name="",
                request_param_value="",
                request_body=request_body,
                expect_body=expect_body
            )
            
        except Exception as e:
            # Fallback to basic test case
            return TestCase(
                test_case_key=f"TC{self.test_case_counter}",
                test_case_name=line,
                prepare_file_name=f"prepareData_Default_{self.test_case_counter}.csv",
                http_method="POST",
                expected_http_code="200",
                request_param_name="",
                request_param_value="",
                request_body="{}",
                expect_body="{}"
            )
    
    def _convert_reference_to_test_case(self, reference_tc: str, index: int) -> Optional[TestCase]:
        """
        Convert reference test case to TestCase object
        
        Args:
            reference_tc: Reference test case string
            index: Index for test case key
            
        Returns:
            TestCase object or None if conversion fails
        """
        try:
            # Try to parse the reference test case
            return self._parse_single_test_case(reference_tc)
        except Exception:
            # If parsing fails, create a basic test case
            return TestCase(
                test_case_key=f"TC{index}",
                test_case_name=reference_tc,
                prepare_file_name=f"prepareData_Reference_{index}.csv",
                http_method="POST",
                expected_http_code="200",
                request_param_name="",
                request_param_value="",
                request_body="{}",
                expect_body="{}"
            )
    
    def _enhance_test_case(self, test_case: TestCase, additional_info: str) -> None:
        """
        Enhance test case with additional information
        
        Args:
            test_case: TestCase object to enhance
            additional_info: Additional information line
        """
        # This method can be used to enhance test cases with additional details
        # For now, it's a placeholder for future enhancement
        pass 