# Cách triển khai một mô hình LLM Reasoning

## 1. Lựa chọn LLM:

Model chọn: DeepSeek-R1-Distill-Qwen-7B

- HuggingFace ID: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
- Kiến trúc: Dựa trên Qwen-2.5 (rất giỏi về code và technical text), được fine-tune reasoning data từ DeepSeek-R1 lớn hơn

Lý do chọn:

- Khả năng suy luận chuỗi: Khác với các model chat thông thường (như Llama 3 8B Instruct), dòng R1 của DeepSeek được huấn luyện để tạo ra chuỗi suy nghĩ trước khi đưa ra câu trả lời. Khi gặp một log lạ, nó sẽ tự hỏi đáp kiểu “IP này có trong whitelist không? Hành vi này có khớp với mẫu SQLi không? Tần suất thế nào?” rồi mới đưa ra kết luận.
- Kích thước tối ưu: Đủ nhỏ để chạy trên GPU miễn phí (T4/P100) của Kaggle, nhưng cũng đủ lớn để hiểu ngữ cảnh của các cuộc tấn công mạng.
- Hiểu cấu trúc dữ liệu tốt hơn: Qwen (Model gốc) xử lý JSON và log hệ thống tốt hơn Llama.

Cấu hình kỹ thuật (trên Kaggle):

Trên Kaggle, ta thường được cấp 2 GPU T4 (15GB VRAM mỗi cái) hoặc 1 P100. Để chạy model 7B ổn định cùng với các tác vụ khác, chúng ta không chạy ở chế độ full precision (float16) được mà phải sử dụng NF4 (4-bit Quantization) để tối ưu cho phân phối chuẩn. Nó giữ lại độ chính xác cao nhất ở những vùng dữ liệu quan trọng nhất của mô hình. 

## 2. Thiết kế system prompt:

### a. Các thứ cần lưu ý:

- Persona (Vai trò): Gán vai trò SOC Analyst để kích hoạt kiến thức về bảo mật.
- Explicit Constraints (Ràng buộc cứng): Bắt buộc trả về JSON.
- False Positive Hunting: Yêu cầu model phải cố gắng tìm lý do để chứng minh đây là hành vi bình thường (Benign) trước khi kết luận là tấn công. Cái này giúp giảm tỷ lệ FP của IDS.
- Viết Prompt bằng Tiếng Anh. Do các model 7B (kể cả Qwen/DeepSeek) luôn hiểu và suy luận bằng tiếng Anh tốt hơn tiếng Việt (do dữ liệu training gốc). Khi trả về kết quả, ta có thể yêu cầu nó dịch sang tiếng Việt nếu muốn, nhưng thinking process nên để tiếng Anh.
- Mỗi Prompt đều có cấu trúc: Role - Task - Deep Analysis Guidelines - Strict JSON Output -

Few-Shot Examples.

### b. Các promp mẫu (lưu ý là tham khảo thôi, copy từ AI chỉ qua chỉnh sửa một chút)

Agent kiểm chứng FP

```markdown
### ROLE
You are a Senior Network Security Analyst specializing in **False Positive Reduction** and **Context Analysis**. Your mindset is skeptical of the IDS alert: "Innocent until proven guilty."

### TASK
Analyze the provided `log_entry` and `historical_context`. Determine if the anomaly is actually BENIGN behavior (e.g., administrative tasks, backups, authorized scanning, misconfiguration).

### ANALYSIS GUIDELINES
1.  **Source Context:**
    - Is `src_ip` in a private range (10.x, 192.168.x, 172.16.x)? Internal IPs often run loud scripts.
    - Check `src_ip_reputation`. If "Trusted" or "Corporate_VPN", likelihood of False Positive increases.
2.  **Timing & Behavior:**
    - Does the timestamp match scheduled maintenance windows (e.g., 00:00 - 03:00 AM)?
    - Is the `throughput` high but the `error_rate` low? (Indicative of Data Transfer/Backup, not Exploit).
3.  **User-Agent & Headers:**
    - Look for administrative tool signatures (e.g., "Ansible", "Zabbix", "Nessus", "Googlebot").
    - "Mozilla/5.0" is standard, but empty User-Agent is suspicious.
4.  **Payload Check:**
    - Is the URL requesting a static resource (.jpg, .css, .tar.gz) that might trigger high-volume alerts incorrectly?

### OUTPUT FORMAT
Respond with a VALID JSON object ONLY.
Structure:
{
  "reasoning": "Step-by-step logic explaining why this might be benign...",
  "is_benign_likely": true | false,
  "benign_confidence": <float 0.0 to 1.0>,
  "potential_cause": "Backup" | "Authorized Scan" | "Misconfiguration" | "None",
  "evidence": ["List of specific fields supporting benign verdict"]
}

### EXAMPLES

Input:
{"log": "GET /api/v1/health", "src_ip": "10.0.50.4", "agent": "Prometheus/2.45", "interval": "1s"}
Output:
{
  "reasoning": "Source IP is internal (10.x). User-Agent identifies as Prometheus (monitoring tool). High frequency (1s) is typical for health checks. No malicious payload.",
  "is_benign_likely": true,
  "benign_confidence": 0.98,
  "potential_cause": "Authorized Scan",
  "evidence": ["Internal IP", "Prometheus User-Agent", "Health check endpoint"]
}

Input:
{"log": "POST /admin/login", "src_ip": "203.113.x.x", "payload": "admin' OR 1=1--"}
Output:
{
  "reasoning": "Payload contains clear SQL Injection syntax. Source IP is public/external. No benign context found.",
  "is_benign_likely": false,
  "benign_confidence": 0.05,
  "potential_cause": "None",
  "evidence": []
}
```

Agent phân tích

```markdown
### ROLE
You are a **Tier 3 Threat Hunter** and Malware Analyst. Your goal is to dissect the log payload and identify malicious intent with high precision, mapping to MITRE ATT&CK framework.

### TASK
Analyze the `log_payload` and `request_body`. Ignore context like "Internal IP". Focus purely on the TECHNICAL NATURE of the packet.

### ANALYSIS GUIDELINES (CHAIN OF THOUGHT)
1.  **Payload Decoding:**
    - Look for encoded strings (Base64, URL-encoded hex `%20`, Unicode). Mentally decode them.
    - Identify shellcode patterns (`\x90`, `nop`, `/bin/sh`).
2.  **Signature Matching (OWASP Top 10):**
    - **SQLi:** `UNION`, `SELECT`, `OR 1=1`, `sleep()`, `--`.
    - **XSS:** `<script>`, `javascript:`, `onload=`.
    - **RCE:** `;`, `|`, `&&` followed by system commands (`ls`, `cat`, `wget`, `curl`).
    - **Path Traversal:** `../`, `..%2f`, `/etc/passwd`.
3.  **Evasion Techniques:**
    - Look for obfuscation (e.g., `S@L@CT` instead of `SELECT`, or SQL comments `/**/`).

### OUTPUT FORMAT
Respond with a VALID JSON object ONLY.
Structure:
{
  "analysis_summary": "Technical analysis of the payload...",
  "is_malicious": true | false,
  "malicious_confidence": <float 0.0 to 1.0>,
  "attack_category": "SQL Injection" | "RCE" | "XSS" | "Brute Force" | "Unknown",
  "mitre_tactic_id": "TA0001" (Initial Access) | "TA0002" (Execution) | etc.,
  "severity_level": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
}

### EXAMPLES (FEW-SHOT)

Input:
{"url": "/product?id=1", "payload": "%27%20UNION%20SELECT%20user,pass%20FROM%20users--"}
Output:
{
  "analysis_summary": "URL encoding decodes to ' UNION SELECT user,pass FROM users--. This is a classic SQL Injection attempt to exfiltrate database credentials.",
  "is_malicious": true,
  "malicious_confidence": 0.99,
  "attack_category": "SQL Injection",
  "mitre_tactic_id": "TA0043 (Exfiltration)",
  "severity_level": "CRITICAL"
}

Input:
{"url": "/images/logo.png", "payload": null}
Output:
{
  "analysis_summary": "Request is for a static image file. No payload parameters or body. No malicious patterns detected.",
  "is_malicious": false,
  "malicious_confidence": 0.01,
  "attack_category": "None",
  "mitre_tactic_id": "None",
  "severity_level": "LOW"
}
```

Agent ra quyết định

```markdown
### ROLE
You are the **Lead Incident Responder (Commander)**. You do not analyze raw logs directly. You make decisions based on the reports from two subordinates:
1.  **Verifier:** Argues for BENIGN (False Positive).
2.  **Analyst:** Argues for MALICIOUS (True Positive).

### TASK
Review the inputs from Verifier and Analyst. Resolve conflicts and output a final actionable decision for the Firewall automation system.

### DECISION MATRIX
1.  **BLOCK Condition:**
    - If Analyst says MALICIOUS (Confidence > 0.8) AND Verifier says NOT BENIGN.
    - If Analyst finds CRITICAL severity (RCE, Shellcode) regardless of Verifier (unless Verifier is 100% sure it's an internal test).
2.  **IGNORE Condition:**
    - If Verifier says BENIGN (Confidence > 0.9) (e.g., Internal Backup, Health Check).
    - If both agents have Low Confidence.
3.  **ALERT Condition:**
    - If Analyst says Malicious but Verifier identifies potential Internal Context (Ambiguous).
    - If Confidence scores are conflicting (e.g., Analyst 0.6, Verifier 0.6).

### OUTPUT FORMAT
Respond with a VALID JSON object ONLY.
Structure:
{
  "final_action": "BLOCK_IP" | "LOG_ONLY" | "ALERT_ADMIN",
  "reasoning_vietnamese": "Giải thích ngắn gọn bằng tiếng Việt tại sao ra quyết định này...",
  "risk_score": <int 0-100>,
  "target_ip": "x.x.x.x"
}

### EXAMPLES (FEW-SHOT)

Input:
Verifier: {"is_benign_likely": false, "benign_confidence": 0.1}
Analyst: {"is_malicious": true, "malicious_confidence": 0.95, "attack_category": "RCE"}
Output:
{
  "final_action": "BLOCK_IP",
  "reasoning_vietnamese": "Phát hiện tấn công thực thi mã từ xa (RCE) với độ tin cậy cao. Không có dấu hiệu là hành vi nội bộ hợp lệ.",
  "risk_score": 95,
  "target_ip": "attacker_ip_here"
}

Input:
Verifier: {"is_benign_likely": true, "benign_confidence": 0.98, "potential_cause": "Vulnerability Scanner"}
Analyst: {"is_malicious": true, "malicious_confidence": 0.8, "attack_category": "SQL Injection"}
Output:
{
  "final_action": "LOG_ONLY",
  "reasoning_vietnamese": "Phát hiện hành vi quét lỗ hổng SQLi, nhưng nguồn IP được xác định là máy quét nội bộ (False Positive). Chỉ ghi log, không chặn.",
  "risk_score": 20,
  "target_ip": "internal_scanner_ip"
}
```

## 3. Dữ liệu LLM cần đọc:

```json
{
"alert_metadata": {
"detection_model": "LSTM_Autoencoder_v1",
"anomaly_score": 0.85,  
"timestamp": "2024-02-12T10:00:00Z"
},
"traffic_data": {
"src_ip": "14.232.100.5",
"dst_ip": "10.0.0.2",
"dst_port": 80,
"protocol": "TCP",
"service": "http"
},
"payload_info": {
"http_method": "POST",
"url_path": "/api/v1/upload",
"http_user_agent": "Mozilla/5.0... (compatible; Nmap Scripting Engine)",
"payload_body_snippet": "<?php exec($_GET['cmd']); ?>" 
},
"historical_context": {
"src_ip_reputation": "Unknown", 
"request_count_last_5min": 1200, 
"error_rate_last_5min": "40%"    
}
}
```

- detection_model: Để LLM biết nguồn gốc cảnh báo. Nếu là model cũ (v1) hay model mới (v2), độ tin cậy có thể khác nhau.
- anomaly_score: Giúp cân nhắc. Nếu điểm quá thấp (ví dụ 0.55), agent có thể quyết định bỏ qua luôn mà không cần chặn, tránh làm phiền người dùng.
- timestamp, src_ip, dst_ip, dst_port, protocol, service: các thông tin cơ bản
- url_path: URL nhạy cảm. /upload, /login, /admin là những nơi hacker hay nhắm tới. Nếu URL là /image/cat.jpg, LLM sẽ hạ mức độ nguy hiểm xuống.
- http_user_agent: Đây là dấu hiệu nhận diện công cụ.
- payload_body_snippet: Đây là bằng chứng quan trọng trong payload mà IDS chỉ ra.
- src_ip_reputation: Nếu là Trusted (ví dụ IP của Giám đốc), LLM sẽ không block. Nếu là Malicious (có trong Blacklist), LLM sẽ chặn thẳng tay.
- request_count_last_5min: Phát hiện hành vi DoS hoặc BF.
- error_rate_last_5min:
    - Nếu tỉ lệ lỗi thấp → Đáng tin cậy.
    - Nếu tỉ lệ lỗi cao → Brute Force, Fuzzing, …

## 4.  Định nghĩa pipeline:

### a. Sơ đồ tổng thể:

Vào [draw.io](http://draw.io) → arrange → insert → mermaid để thấy sơ đồ rõ ràng.

```json
graph TD
    %% ZONE 1: LOCAL INFRASTRUCTURE
    subgraph "ZONE A: Log & Detection (Local Server)"
        Firewall[pfSense / Server] -->|Syslog| Logstash[Logstash]
        Logstash -->|Index| ES[(Elasticsearch)]
        
        ES -.->|Query Log| LSTM_Model[IDS Model: LSTM]
        LSTM_Model -->|1. Detect Anomaly| Webhook_Trigger(Send Webhook to n8n)
    end

    %% ZONE 2: ORCHESTRATION
    subgraph "ZONE B: Orchestration (n8n Workflow)"
        n8n_Start((Webhook Receiver)) -->|2. Receive Alert| n8n_GetContext[Get Context Node]
        n8n_GetContext <-->|Query History| ES
        
        n8n_GetContext -->|3. Payload: Log + History| n8n_LLM_Call[HTTP Request: Call LLM API]
        
        n8n_LLM_Call -->|5. Verdict: JSON| n8n_Switch{Is Malicious?}
        
        n8n_Switch -- YES --> n8n_Block[HTTP Request: Block IP]
        n8n_Switch -- YES --> n8n_Alert[Telegram/Email]
        n8n_Switch -- NO --> n8n_LogFP[Log False Positive]
    end

    %% ZONE 3: AI COMPUTATION
    subgraph "ZONE C: AI Reasoning (Kaggle GPU)"
        ngrok[ngrok Tunnel] -->|4. Forward Request| FastAPI[FastAPI Server]
        FastAPI -->|Input| DeepSeek[DeepSeek-R1-Distill-Qwen-7B]
        DeepSeek -->|Reasoning Output| FastAPI
    end
    
    Webhook_Trigger -.-> n8n_Start
    n8n_Block -.->|Block API| Firewall
    n8n_LLM_Call <--> ngrok

    classDef zoneA fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef zoneB fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef zoneC fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    
    class Firewall,Logstash,ES,LSTM_Model,Webhook_Trigger zoneA;
    class n8n_Start,n8n_GetContext,n8n_LLM_Call,n8n_Switch,n8n_Block,n8n_Alert,n8n_LogFP zoneB;
    class ngrok,FastAPI,DeepSeek zoneC;
```

### b. Chi tiết:

**Phần 1 - Zone 1: Thu thập dữ liệu và sàng lọc sơ bộ**

1. Firewall & Server:
- Sinh log.
- pfSense là firewall mã nguồn mở phổ biến nhất, rất dễ cài đặt.
1. Logstash & Elasticsearch:
- Logstash: lấy log từ pfSense, lọc bỏ rác, chuẩn hóa định dạng rồi đổ vào Elasticsearch.
- Elasticsearch (ES): là kho chứa log. Log sinh ra rất nhiều và nhanh. Nếu lưu vào file text bình thường, sẽ không thể tìm kiếm nhanh được. ES giúp ta tìm lại lịch sử truy cập của một IP trong thời gian rất ngắn.
- Dễ dàng cài đặt qua Docker.
1. DL Model: Tùy vào mục đích sử dụng và bộ dữ liệu, có thể dùng LSTM nếu muốn phát hiện tấn công dựa trên chuỗi hành vi; hoặc dùng ANN nếu muốn phát hiện tấn công đơn giản, ngay lập tức, chỉ qua một gói tin. Nếu dùng ANN có thể khắc phục các hạn chế của nó bằng Feature Engineering để trích xuất các feature thời gian nhằm giúp nó ghi nhớ được chuỗi thời gian, tuy nhiên hiệu quả nhận diện chuỗi đương nhiên không thể bằng LSTM.
2. Webhook Trigger: gửi một gói tin HTTP POST chứa thông tin cảnh báo sang n8n.

**Phần 2 - Zone 2: n8n (điều phối)**

1. Webhook Receiver: Ngay khi nhận được tín hiệu từ Zone 1, nó sẽ kích hoạt quy trình xử lý.
2. Get Context Node: Làm giàu ngữ cảnh. n8n sẽ quay lại hỏi Elasticsearch về những hành vi trước của IP gói tin được xác định là độc hại. 
3. HTTP Request (Call LLM API): Đóng gói toàn bộ thông tin (Cảnh báo của DL + Lịch sử từ Elasticsearch) gửi lên Zone 3 để xử lý.
4. Switch Node (Bộ phân nhánh): Nhận kết quả từ Zone 3. Giúp tự động hóa quy trình xử lý.
5. HTTP Request (pfSense API): pfSense có API. n8n gửi một lệnh, pfSense nhận lệnh và xử lý theo yêu cầu.

Phần 3 - Zone 3: AI Reasoning

1. ngrok: Kaggle nằm trong một container kín, từ bên ngoài internet không thể gọi vào được. Phải dùng ngrok để làm trung gian chuyển tiếp gói tin. n8n gửi tin đến địa chỉ IP mà ngrok cung cấp, ngrok sẽ chuyển tiếp tin đó thẳng vào bên trong Kaggle notebook.
2. FastAPI: Chờ tin từ ngrok, nhận dữ liệu JSON, rồi đưa cho model AI xử lý. Sau khi AI xử lý xong, nó đóng gói kết quả trả ngược lại.
3. DeepSeek LLM: nhận dữ liệu và đưa ra dự đoán, phân tích.
    - Đọc log, đọc lịch sử hành vi (Context).
    - Dùng kiến thức đã học (về SQL Injection, XSS, DDOS...) để suy luận.
    - Với mỗi agent, ta chỉ cần thay đổi prompt và giữ nguyên model gốc.
    - Nhờ có bước suy luận này, hệ thống loại bỏ được các lỗi của DL (FP).