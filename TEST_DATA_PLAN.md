# Test Data Plan for IT Educational Video AI Search Application

## Overview
This document outlines comprehensive test data requirements for positive and negative test scenarios across video, audio, and image collections based on **IT (Information Technology) educational content**. All test data focuses on programming, software engineering, web development, data science, cloud computing, DevOps, cybersecurity, and related IT topics.

### ⏱️ Video Duration Requirements Summary

| Video Type | Duration Range | Purpose |
|------------|---------------|---------|
| **Index Videos** (Production) | **1-5 minutes OR 10 minutes** | Videos to be indexed in the corpus for search functionality |
| **Test Data Videos** | **1-4 minutes** | Test videos used for validation and testing |

**Note**: 
- Index videos should be 1-5 minutes for optimal search granularity, or 10 minutes for comprehensive topics
- All test data videos must be 1-4 minutes for faster test execution and focused validation

---

## 📚 IT Educational Content Categories

### 1. **Programming Languages**
- Python (Basics, Advanced, Frameworks)
- JavaScript (ES6+, Node.js, React, Vue)
- Java (Core Java, Spring Framework, Enterprise)
- C/C++ (System Programming, Algorithms)
- Go, Rust, TypeScript, Kotlin, Swift

### 2. **Web Development**
- Frontend (HTML, CSS, JavaScript, React, Angular, Vue)
- Backend (REST APIs, GraphQL, Serverless)
- Full Stack Development
- Web Security (OWASP, Authentication, Authorization)

### 3. **Data Science & Machine Learning**
- Data Analysis (Pandas, NumPy, Data Visualization)
- Machine Learning (Scikit-learn, TensorFlow, PyTorch)
- Deep Learning (Neural Networks, CNN, RNN, Transformers)
- Data Engineering (ETL, Data Pipelines, Big Data)

### 4. **Cloud & DevOps**
- Cloud Platforms (AWS, Azure, GCP)
- Containerization (Docker, Kubernetes)
- CI/CD (Jenkins, GitHub Actions, GitLab CI)
- Infrastructure as Code (Terraform, Ansible)

### 5. **Software Engineering**
- Software Design Patterns
- System Design & Architecture
- Agile & Scrum Methodologies
- Code Review & Best Practices

### 6. **Cybersecurity**
- Network Security
- Ethical Hacking
- Cryptography
- Security Auditing

### 7. **Database Technologies**
- SQL Databases (PostgreSQL, MySQL, SQL Server)
- NoSQL Databases (MongoDB, Cassandra, Redis)
- Database Design & Optimization

### 8. **Mobile Development**
- iOS Development (Swift, SwiftUI)
- Android Development (Kotlin, Java)
- React Native, Flutter

---

## 🎥 VIDEO TEST DATA

### 📋 Video Duration Requirements

#### **Index Video Requirements**
- **Duration Range**: 1-5 minutes OR 10 minutes
- **Purpose**: Videos to be indexed in the corpus for search
- **Recommended**: Short, focused educational segments (1-5 min) for better indexing and search accuracy
- **Long Format**: 10-minute videos acceptable for comprehensive topics
- **Note**: Videos outside this range may still work but are not optimized

#### **Test Data Requirements**
- **Duration Range**: 1-4 minutes
- **Purpose**: Test videos used to validate search functionality
- **Rationale**: Shorter test videos ensure faster test execution and focused validation
- **Coverage**: All test scenarios should use videos within 1-4 minute range

---

### ✅ Positive Test Scenarios

#### **Valid Educational Videos (Test Data: 1-4 minutes)**

| Test ID | Video Type | Content | Duration | Format | Expected Result |
|---------|-----------|---------|----------|--------|------------------|
| VID-POS-001 | Tutorial | "Python Basics: Variables and Data Types" | 2-3 min | MP4 | Successfully indexed, searchable by text/audio/image |
| VID-POS-002 | Code Walkthrough | "Building REST API with Flask" | 3-4 min | MOV | Indexed with clear timestamps, accurate search results |
| VID-POS-003 | Tutorial | "React Hooks: useState Explained" | 1-2 min | AVI | Visual content indexed, OCR captures code on screen |
| VID-POS-004 | Presentation | "System Design: Microservices Overview" | 4 min | MKV | Video processed in segments, searchable |
| VID-POS-005 | Whiteboard | "Data Structures: Binary Trees Basics" | 2-3 min | MP4 | Handwriting OCR works, visual embeddings captured |
| VID-POS-006 | Screen Recording | "Docker Container Basics" | 3-4 min | MP4 | Code visible, transcript accurate |
| VID-POS-007 | Animation | "How Neural Networks Work" | 2 min | MP4 | Visual embeddings work, diagrams searchable |
| VID-POS-008 | Interview | "Software Engineering Career Tips" | 3-4 min | MP4 | Multiple speakers, accurate transcription |
| VID-POS-009 | Tutorial | "Git Workflow: Branching Strategy" | 2-3 min | MKV | Content searchable, clear explanations |
| VID-POS-010 | Code Review | "Code Review: Python Best Practices" | 1-2 min | MP4 | Code snippets searchable, comments indexed |
| VID-POS-011 | Live Coding | "Building a Todo App with React - Part 1" | 4 min | MP4 | Real-time coding, all code searchable |
| VID-POS-012 | Architecture | "Microservices Architecture Overview" | 3-4 min | MP4 | Diagrams and architecture patterns searchable |
| VID-POS-013 | Security | "Web Security: OWASP Top 10 Overview" | 2-3 min | MP4 | Security concepts and vulnerabilities searchable |
| VID-POS-014 | Database | "SQL JOIN Operations Explained" | 2-3 min | MP4 | SQL queries and concepts indexed |
| VID-POS-015 | Tutorial | "JavaScript Async/Await Basics" | 1-2 min | MP4 | Code examples searchable, clear explanations |

#### **Video Characteristics for Positive Tests**
- **Duration**: **1-4 minutes** (for test data)
- **Resolution**: 720p, 1080p, 4K
- **Frame Rate**: 24fps, 30fps, 60fps
- **Audio**: Clear narration, background music (low), multiple speakers
- **Content**: Slides with text, whiteboard writing, code snippets, diagrams
- **Languages**: English (primary), multilingual with subtitles

#### **Index Video Characteristics (Production Use)**
- **Duration**: **1-5 minutes OR 10 minutes** (recommended for indexing)
- **Resolution**: 720p minimum, 1080p recommended
- **Frame Rate**: 24fps, 30fps
- **Audio**: Clear narration, minimal background noise
- **Content**: Focused educational segments, single topic per video
- **Optimization**: Shorter videos (1-5 min) provide better search granularity

#### **Drive Organization (Positive)**
```
📁 IT_Educational_Videos/
  ├── 📁 Programming_Languages/
  │   ├── 📁 Python/
  │   │   ├── python_basics.mp4
  │   │   ├── python_advanced.mp4
  │   │   └── flask_rest_api.mp4
  │   ├── 📁 JavaScript/
  │   │   ├── javascript_es6.mp4
  │   │   ├── react_tutorial.mp4
  │   │   └── nodejs_backend.mp4
  │   └── 📁 Java/
  │       ├── java_core.mp4
  │       └── spring_framework.mp4
  ├── 📁 Web_Development/
  │   ├── frontend_fundamentals.mp4
  │   ├── backend_development.mp4
  │   └── full_stack_project.mp4
  ├── 📁 Data_Science/
  │   ├── machine_learning_intro.mp4
  │   ├── deep_learning_neural_networks.mp4
  │   └── data_analysis_pandas.mp4
  ├── 📁 Cloud_DevOps/
  │   ├── aws_cloud_services.mp4
  │   ├── docker_kubernetes.mp4
  │   └── cicd_pipelines.mp4
  ├── 📁 Software_Engineering/
  │   ├── design_patterns.mp4
  │   ├── system_design.mp4
  │   └── agile_methodologies.mp4
  ├── 📁 Cybersecurity/
  │   ├── network_security.mp4
  │   └── ethical_hacking.mp4
  └── 📁 Database/
      ├── sql_fundamentals.mp4
      └── nosql_mongodb.mp4
```

### ❌ Negative Test Scenarios

#### **Invalid/Problematic Videos**

| Test ID | Video Type | Issue | Expected Behavior |
|---------|-----------|-------|-------------------|
| VID-NEG-001 | Corrupted File | Video file corrupted | Error message, graceful failure |
| VID-NEG-002 | Unsupported Format | WEBM, FLV, 3GP | Reject with format error message |
| VID-NEG-003 | No Audio Track | Silent video (mute) | Process with visual-only embeddings |
| VID-NEG-004 | Extremely Large | >500MB file | Reject with size limit error |
| VID-NEG-005 | Zero Duration | 0 seconds video | Error: invalid video file |
| VID-NEG-006 | Corrupted Audio | Audio track broken | Fallback to visual-only processing |
| VID-NEG-007 | Very Low Quality | 240p, heavily compressed | Process but warn about quality |
| VID-NEG-008 | No Visual Content | Black screen with audio | Process audio only, warn about visuals |
| VID-NEG-009 | Too Short | <1 minute (for indexing) | Process but warn about optimal duration |
| VID-NEG-010 | Too Long | >10 minutes (for indexing) | Process but warn about optimal duration |
| VID-NEG-011 | Empty File | 0 bytes file | Reject immediately |
| VID-NEG-012 | Wrong MIME Type | Text file renamed to .mp4 | Detect and reject |
| VID-NEG-013 | Password Protected | Encrypted video | Error: cannot process encrypted file |
| VID-NEG-014 | Multiple Audio Tracks | Complex audio structure | Process primary track, warn about others |
| VID-NEG-015 | Rapid Scene Changes | Fast cuts, no stable content | Process but lower confidence scores |
| VID-NEG-016 | Non-Educational | Random content, no structure | Process but may have poor search results |
| VID-NEG-017 | Duration Out of Range | <1 min or >10 min (for indexing) | Process but warn about optimal duration (1-5 or 10 min) |

#### **Drive Organization (Negative)**
```
📁 Test_Negative_Videos/
  ├── corrupted_video.mp4 (corrupted file)
  ├── unsupported_format.webm
  ├── mute_video_no_audio.mp4
  ├── huge_file_600mb.mp4
  ├── zero_duration.mp4
  ├── broken_audio.mp4
  ├── low_quality_240p.mp4
  ├── black_screen.mp4
  ├── very_long_4hours.mp4
  ├── empty_file.mp4
  ├── fake_video.txt (renamed)
  └── encrypted_video.mp4
```

---

## 🎵 AUDIO TEST DATA

### ✅ Positive Test Scenarios

#### **Valid Educational Audio Files**

| Test ID | Audio Type | Content | Duration | Format | Expected Result |
|---------|-----------|---------|----------|--------|------------------|
| AUD-POS-001 | Lecture Audio | "Introduction to Machine Learning" | 45 min | MP3 | Clear transcription, accurate search |
| AUD-POS-002 | Podcast | "Tech Talk: Cloud Computing Trends" | 60 min | WAV | Multiple speakers, accurate timestamps |
| AUD-POS-003 | Interview | "Career Advice for Software Engineers" | 30 min | M4A | Q&A format, searchable by topic |
| AUD-POS-004 | Audiobook | "Clean Code Principles" | 120 min | AAC | Long-form, chapter-based search |
| AUD-POS-005 | Tutorial | "Git and GitHub Workflow" | 20 min | OGG | Clear instructions, searchable commands |
| AUD-POS-006 | Discussion | "Software Architecture Patterns" | 40 min | FLAC | High quality, multiple viewpoints |
| AUD-POS-007 | Code Walkthrough | "Building REST API with Node.js" | 15 min | MP3 | Step-by-step, clear instructions |
| AUD-POS-008 | Presentation | "DevOps Best Practices" | 25 min | WAV | Professional narration |
| AUD-POS-009 | Q&A Session | "Ask Me Anything: Full Stack Development" | 90 min | MP3 | Multiple topics, searchable |
| AUD-POS-010 | Workshop Recording | "React Development Workshop" | 180 min | M4A | Long session, multiple segments |
| AUD-POS-011 | Tech Talk | "Microservices Architecture" | 35 min | MP3 | Technical concepts searchable |
| AUD-POS-012 | Code Review | "Code Review Session: Python Project" | 20 min | WAV | Code discussions indexed |
| AUD-POS-013 | Interview | "Senior Developer Interview Tips" | 45 min | M4A | Career advice searchable |
| AUD-POS-014 | Tutorial | "Docker Containerization Guide" | 30 min | MP3 | Technical steps searchable |
| AUD-POS-015 | Panel Discussion | "Future of Web Development" | 75 min | FLAC | Multiple perspectives searchable |

#### **Audio Characteristics for Positive Tests**
- **Quality**: 128kbps, 192kbps, 320kbps, Lossless
- **Sample Rate**: 44.1kHz, 48kHz
- **Channels**: Mono, Stereo
- **Content**: Clear speech, minimal background noise, structured content
- **Languages**: English (primary), multilingual

#### **Drive Organization (Positive)**
```
📁 IT_Educational_Audio/
  ├── 📁 Lectures/
  │   ├── ml_introduction.mp3
  │   ├── python_basics.mp3
  │   ├── javascript_advanced.mp3
  │   └── system_design.mp3
  ├── 📁 Podcasts/
  │   ├── cloud_computing_trends.wav
  │   ├── ai_ml_discussions.m4a
  │   └── software_engineering_podcast.mp3
  ├── 📁 Interviews/
  │   ├── career_advice_developers.mp3
  │   ├── senior_engineer_interview.aac
  │   └── tech_lead_insights.wav
  ├── 📁 Tutorials/
  │   ├── git_workflow.ogg
  │   ├── docker_tutorial.mp3
  │   └── react_hooks_explained.flac
  └── 📁 Workshops/
      ├── full_stack_workshop.mp3
      └── devops_bootcamp.m4a
```

### ❌ Negative Test Scenarios

#### **Invalid/Problematic Audio Files**

| Test ID | Audio Type | Issue | Expected Behavior |
|---------|-----------|-------|-------------------|
| AUD-NEG-001 | Corrupted File | Audio file corrupted | Error message, graceful failure |
| AUD-NEG-002 | Unsupported Format | WMA, RA, AMR | Reject with format error |
| AUD-NEG-003 | No Audio Content | Silent file | Error: no audio content detected |
| AUD-NEG-004 | Extremely Large | >100MB audio | Reject with size limit error |
| AUD-NEG-005 | Zero Duration | 0 seconds | Error: invalid audio file |
| AUD-NEG-006 | Very Low Quality | 32kbps, heavily compressed | Process but warn about quality |
| AUD-NEG-007 | Heavy Background Noise | Unclear speech | Process but lower accuracy |
| AUD-NEG-008 | Non-Speech | Music only, no speech | Process but may have poor results |
| AUD-NEG-009 | Extremely Long | >4 hours | Process but warn about time |
| AUD-NEG-010 | Empty File | 0 bytes | Reject immediately |
| AUD-NEG-011 | Wrong MIME Type | Video file renamed to .mp3 | Detect and reject |
| AUD-NEG-012 | Encrypted/DRM | Protected audio | Error: cannot process protected file |
| AUD-NEG-013 | Multiple Languages | Rapid language switching | Process but may have mixed results |
| AUD-NEG-014 | Very Fast Speech | Rapid narration | Process but may miss words |
| AUD-NEG-015 | Very Quiet | Low volume, barely audible | Process but warn about quality |

#### **Drive Organization (Negative)**
```
📁 Test_Negative_Audio/
  ├── corrupted_audio.mp3
  ├── unsupported_format.wma
  ├── silent_audio.mp3
  ├── huge_file_150mb.mp3
  ├── zero_duration.mp3
  ├── low_quality_32kbps.mp3
  ├── heavy_noise.mp3
  ├── music_only.mp3
  ├── very_long_5hours.mp3
  ├── empty_file.mp3
  ├── fake_audio.mp4 (renamed)
  └── encrypted_audio.m4a
```

---

## 🖼️ IMAGE TEST DATA

### ✅ Positive Test Scenarios

#### **Valid Educational Images**

| Test ID | Image Type | Content | Resolution | Format | Expected Result |
|---------|-----------|---------|------------|--------|------------------|
| IMG-POS-001 | Architecture Diagram | "Microservices Architecture" | 1920x1080 | PNG | Visual similarity search works |
| IMG-POS-002 | Code Screenshot | "Python REST API Code" | 1920x1080 | PNG | Code readable, visual match works |
| IMG-POS-003 | Database Schema | "ERD: E-commerce Database" | 1600x900 | JPG | Database structure searchable |
| IMG-POS-004 | Flowchart | "CI/CD Pipeline Flowchart" | 2048x1152 | PNG | Process visualization search |
| IMG-POS-005 | Network Diagram | "Cloud Infrastructure Diagram" | 1920x1080 | PNG | Network topology searchable |
| IMG-POS-006 | UI/UX Design | "React Component Design" | 2560x1440 | PNG | Design patterns searchable |
| IMG-POS-007 | Algorithm Visualization | "Binary Tree Structure" | 1920x1080 | PNG | Algorithm visualization search |
| IMG-POS-008 | API Documentation | "REST API Endpoints" | 1200x800 | JPG | API structure searchable |
| IMG-POS-009 | System Design | "Scalable System Architecture" | 1920x1080 | PNG | Architecture patterns searchable |
| IMG-POS-010 | Code Diagram | "Design Patterns: Observer Pattern" | 1600x1200 | PNG | Design pattern visualization |
| IMG-POS-011 | Git Workflow | "Git Branching Strategy" | 1920x1080 | PNG | Workflow diagrams searchable |
| IMG-POS-012 | Data Flow | "Data Pipeline Architecture" | 1920x1080 | PNG | Data flow visualization |
| IMG-POS-013 | Security Diagram | "OAuth 2.0 Flow" | 1600x900 | PNG | Security flow searchable |
| IMG-POS-014 | Deployment Diagram | "Kubernetes Cluster Setup" | 2048x1152 | PNG | Deployment architecture searchable |
| IMG-POS-015 | Code Review Screenshot | "Code Review Comments" | 1920x1080 | PNG | Code review content searchable |

#### **Image Characteristics for Positive Tests**
- **Resolution**: 720p, 1080p, 2K, 4K
- **Format**: JPG, PNG, GIF, WEBP
- **Content**: Clear text, diagrams, charts, educational content
- **Quality**: High quality, minimal compression artifacts
- **Aspect Ratio**: 16:9, 4:3, 1:1

#### **Drive Organization (Positive)**
```
📁 IT_Educational_Images/
  ├── 📁 Architecture_Diagrams/
  │   ├── microservices_architecture.png
  │   ├── system_design_diagram.jpg
  │   └── cloud_infrastructure.png
  ├── 📁 Code_Screenshots/
  │   ├── python_rest_api.png
  │   ├── react_component_code.jpg
  │   └── nodejs_backend.png
  ├── 📁 Database_Schemas/
  │   ├── erd_ecommerce.jpg
  │   └── database_design.png
  ├── 📁 Flowcharts/
  │   ├── cicd_pipeline.png
  │   ├── git_workflow.jpg
  │   └── data_pipeline.png
  ├── 📁 Network_Diagrams/
  │   ├── network_topology.png
  │   └── security_architecture.jpg
  ├── 📁 UI_UX_Designs/
  │   ├── react_component_design.png
  │   └── mobile_app_ui.jpg
  ├── 📁 Algorithm_Visualizations/
  │   ├── binary_tree.png
  │   └── sorting_algorithms.jpg
  └── 📁 Documentation/
      ├── api_endpoints.png
      └── deployment_diagram.jpg
```

### ❌ Negative Test Scenarios

#### **Invalid/Problematic Images**

| Test ID | Image Type | Issue | Expected Behavior |
|---------|-----------|-------|-------------------|
| IMG-NEG-001 | Corrupted File | Image file corrupted | Error message, graceful failure |
| IMG-NEG-002 | Unsupported Format | TIFF, BMP (if not supported) | Reject with format error |
| IMG-NEG-003 | Extremely Large | >50MB image | Reject with size limit error |
| IMG-NEG-004 | Zero Dimensions | 0x0 pixels | Error: invalid image file |
| IMG-NEG-005 | Very Low Resolution | 10x10 pixels | Process but warn about quality |
| IMG-NEG-006 | Extremely High Resolution | >10K pixels | Process but warn about memory |
| IMG-NEG-007 | Empty File | 0 bytes | Reject immediately |
| IMG-NEG-008 | Wrong MIME Type | Video file renamed to .jpg | Detect and reject |
| IMG-NEG-009 | Non-Educational | Random photo, no educational content | Process but may have poor results |
| IMG-NEG-010 | Heavily Compressed | Very low quality, artifacts | Process but warn about quality |
| IMG-NEG-011 | Corrupted Header | Invalid image header | Error: cannot read image |
| IMG-NEG-012 | Animated GIF | Complex animation | Process first frame, warn about animation |
| IMG-NEG-013 | Transparent PNG | Alpha channel issues | Process but may have visual artifacts |
| IMG-NEG-014 | Monochrome | Black and white only | Process but limited visual features |
| IMG-NEG-015 | Watermarked | Heavy watermark overlay | Process but watermark may affect search |

#### **Drive Organization (Negative)**
```
📁 Test_Negative_Images/
  ├── corrupted_image.jpg
  ├── unsupported_format.tiff
  ├── huge_file_60mb.jpg
  ├── zero_dimensions.png
  ├── very_low_res_10x10.jpg
  ├── extremely_high_res_12k.jpg
  ├── empty_file.jpg
  ├── fake_image.mp4 (renamed)
  ├── random_photo.jpg
  ├── heavily_compressed.jpg
  ├── corrupted_header.png
  ├── animated_complex.gif
  └── watermarked.jpg
```

---

## 🔍 SEARCH QUERY TEST DATA

### ✅ Positive Search Scenarios

#### **Text Queries (Positive)**

| Test ID | Query Type | Example Query | Expected Result |
|---------|-----------|---------------|-----------------|
| TXT-POS-001 | Specific Topic | "What is dependency injection?" | Returns relevant segments about DI in software development |
| TXT-POS-002 | Concept Explanation | "Explain neural networks" | Returns ML/AI educational segments |
| TXT-POS-003 | How-to Question | "How to build a REST API with Flask?" | Returns tutorial segments on Flask API |
| TXT-POS-004 | Definition | "What is microservices architecture?" | Returns definition and explanation segments |
| TXT-POS-005 | Comparison | "Difference between REST and GraphQL" | Returns comparison segments |
| TXT-POS-006 | Code Search | "Python list comprehension example" | Returns code demonstration segments |
| TXT-POS-007 | Framework | "React hooks useState useEffect" | Returns React tutorial segments |
| TXT-POS-008 | Algorithm | "Binary search algorithm implementation" | Returns algorithm explanation segments |
| TXT-POS-009 | Process | "How does CI/CD pipeline work?" | Returns DevOps process explanation segments |
| TXT-POS-010 | Multi-word | "machine learning supervised learning algorithms" | Returns relevant ML segments |
| TXT-POS-011 | Database | "SQL JOIN operations explained" | Returns database tutorial segments |
| TXT-POS-012 | Security | "OWASP top 10 vulnerabilities" | Returns security educational segments |
| TXT-POS-013 | Cloud | "AWS S3 bucket configuration" | Returns cloud computing segments |
| TXT-POS-014 | Design Pattern | "Observer pattern implementation" | Returns design pattern segments |
| TXT-POS-015 | System Design | "How to design a scalable system?" | Returns system design segments |

#### **Audio Query Queries (Positive)**

| Test ID | Audio Content | Expected Result |
|---------|--------------|-----------------|
| AUDQ-POS-001 | Question: "What is machine learning?" | Finds segments explaining ML concepts |
| AUDQ-POS-002 | Statement: "Let's learn about Python programming" | Finds Python tutorial segments |
| AUDQ-POS-003 | Multiple Topics: "Today we'll cover React hooks and state management" | Finds React-related segments |
| AUDQ-POS-004 | Code Explanation: "Here's how to use async await in JavaScript" | Finds code example segments |
| AUDQ-POS-005 | Technical Explanation: "Docker containers provide isolation..." | Finds Docker/containerization segments |
| AUDQ-POS-006 | Framework Discussion: "Spring Framework dependency injection" | Finds Spring Framework segments |
| AUDQ-POS-007 | Architecture: "Microservices architecture benefits" | Finds architecture discussion segments |
| AUDQ-POS-008 | Database: "SQL JOIN operations and their types" | Finds database tutorial segments |

#### **Image Query Queries (Positive)**

| Test ID | Image Content | Expected Result |
|---------|--------------|-----------------|
| IMGQ-POS-001 | Microservices architecture diagram | Finds similar architecture diagrams in videos |
| IMGQ-POS-002 | Python code screenshot | Finds similar code examples in videos |
| IMGQ-POS-003 | Database ERD diagram | Finds similar database design segments |
| IMGQ-POS-004 | CI/CD pipeline flowchart | Finds similar DevOps process visualizations |
| IMGQ-POS-005 | React component design | Finds similar UI/component discussions |
| IMGQ-POS-006 | Network topology diagram | Finds similar network architecture segments |
| IMGQ-POS-007 | Algorithm flowchart | Finds similar algorithm explanation segments |
| IMGQ-POS-008 | API endpoint documentation | Finds similar API tutorial segments |

### ❌ Negative Search Scenarios

#### **Text Queries (Negative)**

| Test ID | Query Type | Example Query | Expected Behavior |
|---------|-----------|---------------|-------------------|
| TXT-NEG-001 | Empty Query | "" | Error: query cannot be empty |
| TXT-NEG-002 | Very Long Query | 1000+ characters | Process but warn about length |
| TXT-NEG-003 | Special Characters Only | "!@#$%^&*()" | Process but may return no results |
| TXT-NEG-004 | Non-IT Content | "What's the weather today?" | Returns no relevant IT results |
| TXT-NEG-005 | Offensive Content | [Inappropriate query] | Filter and reject |
| TXT-NEG-006 | Non-English (if not supported) | "¿Qué es Python?" | Process if multilingual supported |
| TXT-NEG-007 | SQL Injection Attempt | "'; DROP TABLE--" | Sanitize and process safely |
| TXT-NEG-008 | XSS Attempt | "<script>alert('xss')</script>" | Sanitize and process safely |

#### **Audio Query Queries (Negative)**

| Test ID | Audio Issue | Expected Behavior |
|---------|------------|-------------------|
| AUDQ-NEG-001 | No Speech | Music only | Process but may return poor results |
| AUDQ-NEG-002 | Unclear Audio | Heavy noise | Process but lower accuracy |
| AUDQ-NEG-003 | Wrong Language | Non-English (if not supported) | Process if supported, else error |
| AUDQ-NEG-004 | Corrupted Audio | Broken file | Error: cannot process audio |

#### **Image Query Queries (Negative)**

| Test ID | Image Issue | Expected Behavior |
|---------|------------|-------------------|
| IMGQ-NEG-001 | Non-Educational | Random photo | Process but may return poor results |
| IMGQ-NEG-002 | Very Low Quality | Blurry, pixelated | Process but lower accuracy |
| IMGQ-NEG-003 | Corrupted Image | Broken file | Error: cannot process image |
| IMGQ-NEG-004 | Empty/Black Image | No content | Process but may return no results |

---

## 📊 TEST DATA SUMMARY TABLE

### Collection Requirements

| Media Type | Positive Tests | Negative Tests | Total Files | Drive Size Estimate |
|------------|---------------|---------------|-------------|---------------------|
| **Videos** | 15 files (1-4 min) | 17 files | 32 files | ~5-8 GB |
| **Audio** | 15 files | 15 files | 30 files | ~3-4 GB |
| **Images** | 15 files | 15 files | 30 files | ~1-2 GB |
| **Search Queries** | 15 queries | 12 queries | 27 queries | N/A |
| **TOTAL** | 60 items | 59 items | **119 items** | **~9-14 GB** |

---

## 🗂️ RECOMMENDED DRIVE STRUCTURE

```
📁 VideoAI_IT_TestData/
│
├── 📁 01_Positive_Tests/
│   ├── 📁 Videos/
│   │   ├── 📁 Programming_Languages/
│   │   ├── 📁 Web_Development/
│   │   ├── 📁 Data_Science/
│   │   ├── 📁 Cloud_DevOps/
│   │   ├── 📁 Software_Engineering/
│   │   ├── 📁 Cybersecurity/
│   │   └── 📁 Database/
│   ├── 📁 Audio/
│   │   ├── 📁 Lectures/
│   │   ├── 📁 Podcasts/
│   │   ├── 📁 Interviews/
│   │   ├── 📁 Tutorials/
│   │   └── 📁 Workshops/
│   └── 📁 Images/
│       ├── 📁 Architecture_Diagrams/
│       ├── 📁 Code_Screenshots/
│       ├── 📁 Database_Schemas/
│       ├── 📁 Flowcharts/
│       ├── 📁 Network_Diagrams/
│       ├── 📁 UI_UX_Designs/
│       ├── 📁 Algorithm_Visualizations/
│       └── 📁 Documentation/
│
├── 📁 02_Negative_Tests/
│   ├── 📁 Videos/
│   ├── 📁 Audio/
│   └── 📁 Images/
│
├── 📁 03_Search_Queries/
│   ├── 📁 Text_Queries/
│   │   ├── programming_queries.txt
│   │   ├── framework_queries.txt
│   │   ├── architecture_queries.txt
│   │   └── devops_queries.txt
│   ├── 📁 Audio_Queries/
│   └── 📁 Image_Queries/
│
└── 📁 04_Documentation/
    ├── TEST_DATA_PLAN.md (this file)
    ├── test_results_template.xlsx
    └── test_execution_log.md
```

---

## ✅ TEST EXECUTION CHECKLIST

### Pre-Test Setup
- [ ] Create drive folder structure
- [ ] Upload all positive test files (1-4 minute videos)
- [ ] Upload all negative test files
- [ ] Prepare search query files
- [ ] Verify file formats and sizes
- [ ] Verify video durations (test data: 1-4 min, index videos: 1-5 or 10 min)
- [ ] Document file metadata

### Positive Test Execution
- [ ] Test video indexing (all 10 positive videos)
- [ ] Test audio file processing (all 10 positive audio)
- [ ] Test image file processing (all 10 positive images)
- [ ] Test text search queries (all 15 positive queries)
- [ ] Test audio search queries (all 5 positive queries)
- [ ] Test image search queries (all 5 positive queries)
- [ ] Verify search result accuracy
- [ ] Test highlight reel generation

### Negative Test Execution
- [ ] Test corrupted files (all media types)
- [ ] Test unsupported formats
- [ ] Test size limit violations
- [ ] Test empty files
- [ ] Test invalid queries
- [ ] Verify error handling
- [ ] Verify error messages are user-friendly

### Post-Test
- [ ] Document all test results
- [ ] Create bug reports for failures
- [ ] Update test data based on findings
- [ ] Archive test results

---

## 📝 NOTES

1. **File Naming Convention**: Use descriptive names with test IDs
   - Example: `VID-POS-001_python_intro.mp4`
   - Example: `AUD-NEG-003_silent_audio.mp3`

2. **Metadata Tracking**: Maintain a spreadsheet with:
   - File name, test ID, expected result, actual result, notes

3. **Version Control**: Keep test data in version control or document changes

4. **Privacy**: Ensure educational content used is either:
   - Public domain
   - Licensed for testing
   - Created specifically for testing

5. **Storage**: Consider cloud storage (Google Drive, Dropbox) for easy access

---

## 🔄 UPDATES AND MAINTENANCE

- Update test data as application features evolve
- Add new test cases based on discovered bugs
- Remove obsolete test cases
- Document any changes to test data structure

---

**Last Updated**: 2025-11-28  
**Version**: 1.0  
**Maintained By**: Development Team

