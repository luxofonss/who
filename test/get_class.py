import re
import json
from typing import List, Dict, Set
import logging
logger = logging.getLogger(__name__)

def extract_changed_methods(diff_text: str) -> List[Dict[str, str]]:
     """
     Extract all methods that have changes in a git diff.
     Returns a list of strings with both class names and class.element format
     """
     changed_methods = []
     
     # Split diff into file sections
     file_sections = re.split(r'^diff --git', diff_text, flags=re.MULTILINE)
     
     for section in file_sections:
          if not section.strip():
               continue
               
          # Extract file path and class name
          file_path = _extract_file_path(section)
          if not file_path or not file_path.endswith('.java'):
               continue
               
          class_name = _extract_class_name(section, file_path)
          methods = _find_changed_methods_in_section(section, class_name)
          logger.info(f"Methods: {methods}")
          
          changed_methods.append({"class": class_name, "method": None})
          
          # Add all the specific method/field changes
          changed_methods.extend(methods)
     
     return sorted(list(changed_methods), key=lambda x: x["class"])

def _extract_file_path(section: str) -> str:
     """Extract file path from diff section header"""
     # Look for the +++ line which contains the new file path
     for line in section.split('\n'):
          if line.startswith('+++'):
               # Remove +++ and any a/ or b/ prefix
               path = line[4:].strip()
               if path.startswith('a/') or path.startswith('b/'):
                    path = path[2:]
               return path
     return ""

def _extract_class_name(section: str, file_path: str) -> str:
     """Extract class name from diff section or file path"""
     # First try to find class declaration in the diff
     class_pattern = re.compile(r'^\s*(?:public\s+)?(?:final\s+)?(?:class|interface|record)\s+(\w+)', re.MULTILINE)
     
     for line in section.split('\n'):
          # Look in both added and removed lines, and context lines
          clean_line = line[1:] if line.startswith(('+', '-', ' ')) else line
          match = class_pattern.match(clean_line.strip())
          if match:
               return match.group(1)
     
     # Fallback to filename
     return file_path.split('/')[-1].replace('.java', '')

def _find_changed_methods_in_section(section: str, class_name: str) -> List[Dict[str, str]]:
     """Find all methods that have changes in this file section"""
     changed_methods = []
     
     # Pattern to match method signatures
     method_pattern = re.compile(
          r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?'
          r'(?:synchronized\s+)?(?:abstract\s+)?'
          r'(?:<[^>]+>\s+)?'  # Generic type parameters
          r'(?:[\w\[\]<>.,\s]+\s+)'  # Return type
          r'(\w+)\s*\('  # Method name
     )
     
     lines = section.split('\n')
     current_context = None
     current_method = None
     brace_depth = 0
     
     for i, line in enumerate(lines):
          # Skip file header lines
          if line.startswith(('diff --git', 'index', '---', '+++')):
               continue
               
          # Check if this is a hunk header
          if line.startswith('@@'):
               current_method = None
               current_context = None
               brace_depth = 0
               continue
          
          # Only process lines that are part of the diff (added, removed, or context)
          if not line.startswith(('+', '-', ' ')):
               continue
               
          clean_line = line[1:] if line.startswith(('+', '-', ' ')) else line
          
          # Update brace depth to track context
          brace_depth += clean_line.count('{') - clean_line.count('}')
          
          # Check if this line contains a method signature
          method_match = method_pattern.match(clean_line.strip())
          if method_match:
               method_name = method_match.group(1)
               current_method = method_name
               current_context = 'method'
               
               # If this is a changed line (method signature changed), mark it
               if line.startswith(('+', '-')):
                    changed_methods.append({
                    "class": class_name,
                    "method": method_name
               })
          
          # If we see a change line, determine what changed
          elif line.startswith(('+', '-')):
               clean_stripped = clean_line.strip()
               
               # Skip empty lines and pure structural changes
               if not clean_stripped or clean_stripped in ['{', '}']:
                    continue
               
               # If we're in a method, attribute change to that method
               if current_method and brace_depth > 0:
                    changed_methods.append({
                    "class": class_name,
                    "method": current_method
               })
               else:
               # This is a class-level change (field, annotation, etc.)
               # Look for the nearest field or method to attribute it to
                    nearest_element = _find_nearest_element(lines, i, class_name)
                    if nearest_element:
                         changed_methods.append({
                              "class": class_name,
                              "method": nearest_element
                         })
                    else:
                         # If no specific element found, mark as class-level change
                         changed_methods.append({
                              "class": class_name,
                              "method": None
                         })
          
          # Update current method context based on brace depth
          if current_method and brace_depth == 0:
               current_method = None
               current_context = None

     return changed_methods

def _find_nearest_element(lines: List[str], change_idx: int, class_name: str) -> str:
     """Find the nearest field or method to attribute a class-level change to"""
     
     # Patterns to match fields and methods
     field_pattern = re.compile(r'^\s*(?:private|public|protected)?\s*(?:static\s+)?(?:final\s+)?[\w\[\]<>.,\s]+\s+(\w+)\s*[;=]')
     method_pattern = re.compile(r'^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?(?:abstract\s+)?(?:<[^>]+>\s+)?(?:[\w\[\]<>.,\s]+\s+)(\w+)\s*\(')
     
     # Look forward and backward from the change
     for offset in range(1, min(10, len(lines) - change_idx)):
          # Check lines after the change
          if change_idx + offset < len(lines):
               line = lines[change_idx + offset]
               if line.startswith(('+', '-', ' ')):
                    clean_line = line[1:] if line.startswith(('+', '-', ' ')) else line
               
               # Try to match field
               field_match = field_pattern.match(clean_line.strip())
               if field_match:
                    return field_match.group(1)  # Return only the field name
               
               # Try to match method
               method_match = method_pattern.match(clean_line.strip())
               if method_match:
                    return method_match.group(1)  # Return only the method name
          
          # Check lines before the change
          if change_idx - offset >= 0:
               line = lines[change_idx - offset]
               if line.startswith(('+', '-', ' ')):
                    clean_line = line[1:] if line.startswith(('+', '-', ' ')) else line
               
               # Try to match field
               field_match = field_pattern.match(clean_line.strip())
               if field_match:
                    return field_match.group(1)  # Return only the field name
               
               # Try to match method
               method_match = method_pattern.match(clean_line.strip())
               if method_match:
                    return method_match.group(1)  # Return only the method name
     
     return ""
# ==== HARD-CODED TEST INPUT ====
diff_text = """diff --git a/src/main/java/com/edu/onestudy/controller/external/QuizController.java b/src/main/java/com/edu/onestudy/controller/external/QuizController.java
index 4281bb4..5b36755 100644
--- a/src/main/java/com/edu/onestudy/controller/external/QuizController.java
+++ b/src/main/java/com/edu/onestudy/controller/external/QuizController.java
@@ -125,6 +125,13 @@ public class QuizController {
     @GetMapping("/search")
     @LogsActivityAnnotation
     BaseResponse<List<Quiz>> search(@RequestParam String query) {
-        return quizService.search(query);
+        return baseService.ofSucceeded(quizService.search(query));
     }
+
+    @GetMapping("/search2")
+    @LogsActivityAnnotation
+    BaseResponse<List<Quiz>> search2(@RequestParam String query) {
+        return baseService.ofSucceeded(quizService.search2(query));
+    }
+
 }
diff --git a/src/main/java/com/edu/onestudy/service/QuizService.java b/src/main/java/com/edu/onestudy/service/QuizService.java
index e6b8ec8..3076cba 100644
--- a/src/main/java/com/edu/onestudy/service/QuizService.java
+++ b/src/main/java/com/edu/onestudy/service/QuizService.java
@@ -46,4 +46,6 @@ public interface QuizService {
     List<UserSavedQuiz> getUserSavedQuiz(String quizId, UUID userId);
 
     List<Quiz> search(String query);
+
+    List<Quiz> search2(String query);
 }
diff --git a/src/main/java/com/edu/onestudy/service/impl/QuizServiceImpl.java b/src/main/java/com/edu/onestudy/service/impl/QuizServiceImpl.java    
index 7fd3f63..096d45e 100644
--- a/src/main/java/com/edu/onestudy/service/impl/QuizServiceImpl.java
+++ b/src/main/java/com/edu/onestudy/service/impl/QuizServiceImpl.java
@@ -513,6 +513,13 @@ public class QuizServiceImpl implements QuizService {
         return quizRepository.findAll();
     }
 
+    @Override
+    public List<Quiz> search2(String query) {
+        if (query == null) throw new BusinessException(ErrorConstant.QUIZ_NOT_FOUND); //TODO: fix msg
+
+        return quizRepository.findAll();
+    }
+
     private void getQuizAuthor(List<Quiz> quizzes) {
         quizzes.forEach(q -> {
             if (q.getAuthorId() != null) {
"""

file_path = "src/main/java/com/edu/onestudy/dto/quiz/CreateQuizRequest.java"

# ==== EXTRACT AND WRITE TO FILE ====
results = extract_changed_methods(diff_text)

with open("changed_elements.json", "w") as f:
    json.dump(results, f, indent=2)

print("Extraction complete. Results written to changed_elements.json")
