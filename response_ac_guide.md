{
  "analysis_metadata": { // Metadata về quá trình phân tích
    "jira_task_id": "ID của task Jira liên quan, vd: SCRUM-1",
    "git_commit_id": "List ID của commit Git đã được phân tích, có thể có nhiều diff git, vd: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
    "analysis_date": "Ngày và giờ phân tích được thực hiện (ISO 8601), vd: 2025-07-10T15:13:30+07:00"
  },
  "detailed_mapping": [ // Danh sách chi tiết các tiêu chí chấp nhận (AC) và phân tích tương ứng
    {
      "id": "Mã định danh duy nhất cho tiêu chí chấp nhận (ví dụ: AC1, AC2, AC_AI_Suggest1), vd: AC1",
      "ac_description": "Mô tả chi tiết của tiêu chí chấp nhận (AC) hoặc yêu cầu. Nội dung này được tổng hợp từ hai nguồn chính:1.Các yêu cầu được trích xuất trực tiếp từ tài liệu Jira: Đây là những tiêu chí đã được định nghĩa rõ ràng, là cơ sở ban đầu của tính năng.2.Các yêu cầu mới được hệ thống tự động đề xuất: Dựa trên quá trình phân tích chuyên sâu, hệ thống sẽ tạo ra các AC bổ sung nhằm nâng cao tính đầy đủ, độ ổn định và trải nghiệm người dùng tổng thể của tính năng. Các đề xuất này được xây dựng dựa trên:2.1.Các chuẩn công nghiệp và nguyên tắc thiết kế API tốt nhất: Bao gồm RESTful principles, xử lý mã trạng thái HTTP chuẩn, quản lý lỗi (error handling) nhất quán, xác thực đầu vào (input validation) mạnh mẽ, phân trang (pagination), lọc (filtering), sắp xếp (sorting), và các cân nhắc về bảo mật (authentication, authorization).2.2.Phân tích dữ liệu lịch sử và mô hình từ các API tương tự: Hệ thống học hỏi từ các API backend đã được phát triển trước đó trong cùng hệ sinh thái hoặc các dự án tương tự. Điều này bao gồm việc nhận diện các trường hợp sử dụng phổ biến, các kịch bản lỗi thường gặp, và các tính năng bổ trợ (ví dụ: tìm kiếm, phân tích log, monitoring) mà một API có chức năng tương tự thường yêu cầu.2.3.Các kịch bản sử dụng và trường hợp biên tiềm năng: Dựa trên mô tả tính năng, hệ thống sẽ chủ động suy luận các tình huống mà người dùng có thể gặp phải hoặc các điều kiện dữ liệu đặc biệt (ví dụ: dữ liệu rỗng, dữ liệu quá lớn, giá trị âm/dương ngoài phạm vi, tấn công Injection) để đề xuất các AC kiểm tra tính bền vững và an toàn của API.",
      "status": "Trạng thái của AC: 'Đã định nghĩa', 'Chưa định nghĩa', 'Cần làm rõ'. Cho biết nguồn gốc và trạng thái hiện tại của tiêu chí chấp nhận. Đây là chỉ số quan trọng cho BA:"Đã định nghĩa": Tiêu chí chấp nhận này đã được nêu rõ ràng trong các tài liệu yêu cầu chính thức (ví dụ: Jira)."Chưa định nghĩa": Tiêu chí chấp nhận này được hệ thống AI đề xuất bổ sung dựa trên phân tích để đảm bảo tính năng hoàn chỉnh, bao quát các trường hợp biên hoặc cải tiến tiềm năng. BA cần xem xét để định nghĩa thêm."Cần làm rõ": Yêu cầu này cần được trao đổi thêm với các bên liên quan (ví dụ: BA/PO) để làm rõ. Tình trạng này cũng đặc biệt quan trọng khi Test case đã tồn tại nhưng chưa có yêu cầu rõ ràng, hoặc Code đã được viết nhưng chưa được định nghĩa trong yêu cầu. BA cần xác định lại hoặc bổ sung yêu cầu.",
      "testcase_name": "Tên hoặc mô tả ngắn gọn của trường hợp kiểm thử liên quan đến tiêu chí chấp nhận. Nếu test case có sẵn trong tài liệu, nội dung gốc sẽ được giữ. Nếu là đề xuất mới, nó sẽ tuân theo định dạng chuẩn của hướng dẫn kiểm thử. Đây là thông tin cốt lõi cho Tester.",
      "code_location": "Vị trí cụ thể trong mã nguồn (ví dụ: TênClass.java (:SốDòng) hoặc tên class / method) nơi tiêu chí chấp nhận này được triển khai. Nếu không có phần code trực tiếp nào liên quan rõ ràng, trường này sẽ để trống. Đây là thông tin cốt lõi cho Dev",
      "assessment": "Đánh giá cụ thể về mức độ tiêu chí chấp nhận này đã được đáp ứng. Đây là chỉ số trực tiếp cho cả BA, Dev, và Tester về tình trạng "thừa/thiếu":"Đạt yêu cầu": Tiêu chí chấp nhận đã được triển khai hoàn chỉnh trong mã nguồn và có trường hợp kiểm thử đầy đủ bao phủ. (Mọi vai trò đều hài lòng)."Chưa có code": Tiêu chí chấp nhận này chưa có bất kỳ mã nguồn triển khai nào. (Dev cần hành động)."Chưa có testcase": Tiêu chí chấp nhận này chưa có trường hợp kiểm thử nào được xác định hoặc triển khai. (Tester cần hành động)."Testcase chưa đủ": Trường hợp kiểm thử hiện tại chỉ bao phủ một phần của tiêu chí chấp nhận hoặc chưa đủ để kiểm tra toàn bộ các khía cạnh cần thiết. (Tester cần bổ sung)."Code chưa đủ": Mã nguồn hiện tại chỉ triển khai một phần của tiêu chí chấp nhận hoặc chưa đủ để đáp ứng đầy đủ các yêu cầu. (Dev cần hoàn thiện)."BA cần xem xét": Tiêu chí chấp nhận hoặc kịch bản liên quan cần được Business Analyst (BA) xem xét và làm rõ thêm trước khi tiếp tục triển khai hoặc kiểm thử. (Có thể là Test case đã có nhưng chưa có yêu cầu, hoặc Code đã viết nhưng yêu cầu không rõ ràng)"
      "priority": "Mức độ ưu tiên để hành động: 'High', 'Medium', 'Low', 'N/A' (nếu đã 'Đạt yêu cầu'), vd: N/A"
    }
  ],
  "coverage_overview": { // Tổng quan về mức độ bao phủ và chất lượng tổng thể
    "total_acceptance_criteria": "Tổng số lượng các tiêu chí chấp nhận (AC) đã xác định, vd: 9",
    "fully_covered": "Số lượng AC có assessment là 'Đạt yêu cầu', vd: 5",
    "partially_covered": "Số lượng AC có assessment là 'Testcase chưa đủ' hoặc 'Code chưa đủ', vd: 2",
    "not_covered": "Số lượng AC có assessment là 'Chưa có code' hoặc 'Chưa có testcase', vd: 2",
    "requirement_coverage": "Tỷ lệ % AC có status 'Đã định nghĩa' trên tổng số AC, vd: 56%",
    "code_coverage": "Tỷ lệ % AC có status 'Đã định nghĩa' và assessment là 'Đạt yêu cầu' hoặc 'Code chưa đủ' trên tổng số AC 'Đã định nghĩa', vd: 60%",
    "test_case_coverage": "Tỷ lệ % AC có status 'Đã định nghĩa' và assessment là 'Đạt yêu cầu' hoặc 'Testcase chưa đủ' trên tổng số AC 'Đã định nghĩa', vd: 60%",
    "assessment": "Đánh giá tổng thể về sự hoàn thiện và chất lượng ('Satisfactory' / 'Not Satisfactory'), vd: Not Satisfactory",
    "quality_score": "Điểm chất lượng tổng thể (trên thang 100), được tính toán dựa trên các chỉ số bao phủ, vd: 75",
    "visual_summary_data": { // Dữ liệu thô để phục vụ việc vẽ biểu đồ trực quan trên giao diện người dùng
      "coverage_distribution": { // Dữ liệu phân phối cho biểu đồ tròn (Pie chart)
        "fully_covered": "Số lượng AC được bao phủ hoàn toàn, vd: 5",
        "partially_covered": "Số lượng AC được bao phủ một phần, vd: 2",
        "not_covered": "Số lượng AC chưa được bao phủ, vd: 2"
      },
      "progress_metrics": { // Dữ liệu tiến độ cho thanh hiển thị (Progress bars)
        "requirement": "Giá trị phần trăm cho độ bao phủ yêu cầu, vd: 56",
        "code": "Giá trị phần trăm cho độ bao phủ mã nguồn, vd: 60",
        "test_case": "Giá trị phần trăm cho độ bao phủ test case, vd: 60"
      }
    }
  }
}