# 项目分析请求
_Exported on 2025/12/12 at GMT+8 19:12:34 from Cursor (2.1.32)_

---

**User**

分析该项目

---

**Cursor**

[MODE: RESEARCH]

我将开始分析这个项目的结构和内容。首先让我查看项目的根目录结构。

嗯... 通过深入查看项目的目录结构、核心代码和配置文件，我已经对该项目有了系统性的理解。这是一个相当复杂和完整的端侧语音交互系统，从技术架构、实现方式到应用场景都很有价值。让我为您详细分析这个项目。

## 项目整体概览

这是一个**边缘端侧部署的全离线智能语音交互系统**，主要针对车载座舱场景。项目包含三个主要子系统：

1. **Edge_LLM_RAG_Voice** - 带RAG知识库的完整语音交互系统（最核心）
2. **LLM_Voice_Flow** - 基础语音交互流水线系统
3. **Edge-LLM-Infra** - 边缘端基础设施和通信框架

## 核心技术架构分析

### 1. 系统架构特点

从代码中可以看出，这是一个**模块化、松耦合**的系统设计：

#### 核心模块组成：
- **ASR模块（语音识别）** - 基于sherpa-onnx实现流式语音识别
- **RAG知识检索模块** - 车载手册的向量搜索系统
- **LLM推理模块** - 基于RK3576 NPU的DeepSeek大模型推理
- **TTS模块（语音合成）** - 基于自研引擎的语音合成系统

#### 通信机制：
- 使用**ZeroMQ**实现模块间通信
- 采用REQ/REP和PUSH/PULL模式
- 各模块通过标准化的接口交互

### 2. RAG知识检索系统分析

这是项目的核心创新点，查看`EdgeLLMRAGSystem`类：

```cpp:1:68:Edge_LLM_RAG_Voice/automotive_edge_rag/cpp/edge_llm_rag_system.h
#pragma once

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>

#include <unordered_map>
#include <atomic>
#include "query_classifier.h"
#include "ZmqServer.h"
#include "ZmqClient.h"

namespace fs = std::filesystem;

namespace py = pybind11;

namespace edge_llm_rag
{

    class EdgeLLMRAGSystem
    {
    public:
        explicit EdgeLLMRAGSystem();
        ~EdgeLLMRAGSystem();

        bool initialize();

        std::string process_query(const std::string &query,
                                  const std::string &user_id = "",
                                  const std::string &context = "");

        QueryClassification classify_query(const std::string &query);

        std::string rag_only_response(const std::string &query, bool preload = false);

        std::string llm_only_response(const std::string &query);

        std::string hybrid_response(const std::string &query);

        bool cleanup_cache();

    private:
        bool is_initialized_;

        py::object searcher;
        py::scoped_interpreter guard{};

        zmq_component::ZmqClient tts_client_{"tcp://localhost:7777"};
        zmq_component::ZmqClient llm_client_{"tcp://localhost:8899"};

        std::unique_ptr<QueryClassifier>
            query_classifier_;

        std::unordered_map<std::string, std::string> query_cache_;

        bool add_to_cache(const std::string &query, const std::string &response);
        std::string get_from_cache(const std::string &query);
        bool is_cache_valid(const std::string &query);

        void rag_message_worker(const std::string &rag_text);
        bool preload_common_queries();
    };

} // namespace edge_llm_rag
```

#### 核心特性：

**多级响应策略** - 这是项目的一大亮点，从`process_query`方法可以看到：

```cpp:88:120:Edge_LLM_RAG_Voice/automotive_edge_rag/cpp/edge_llm_rag_system.cpp
        switch (classification.query_type)
        {
        case QueryClassification::EMERGENCY_QUERY:
            std::cout << "===============================" << std::endl;
            std::cout << "紧急查询 detected, using RAG only response." << std::endl;
            std::cout << "===============================" << std::endl;
            response = rag_only_response(query);
            break;
        case QueryClassification::FACTUAL_QUERY:
            std::cout << "===============================" << std::endl;
            std::cout << "事实性查询 detected, using RAG only response." << std::endl;
            std::cout << "===============================" << std::endl;
            response = rag_only_response(query);
            break;
        case QueryClassification::COMPLEX_QUERY:
            std::cout << "===============================" << std::endl;
            std::cout << "复杂查询 detected, using hybrid response." << std::endl;
            std::cout << "===============================" << std::endl;

            response = hybrid_response(query);
            break;
        case QueryClassification::CREATIVE_QUERY:
            std::cout << "===============================" << std::endl;
            std::cout << "创意查询 detected, using LLM only response." << std::endl;
            std::cout << "===============================" << std::endl;
            response = llm_only_response(query);
            break;
        default:
            std::cout << "===============================" << std::endl;
            std::cout << "未知查询类型, using adaptive response." << std::endl;
            std::cout << "===============================" << std::endl;
            response = hybrid_response(query);
        }
```

系统根据查询类型采用不同的响应策略：
- **紧急查询/事实性查询** → 仅RAG检索（最快响应）
- **复杂查询** → RAG + LLM混合模式（精准度优先）
- **创意查询** → 仅LLM（灵活性优先）

**智能查询分类器** - 通过关键词和特征分析：

```cpp:11:39:Edge_LLM_RAG_Voice/automotive_edge_rag/cpp/query_classifier.h
    struct QueryFeatures
    {
        std::vector<std::string> keywords;
        float urgency_score;               
        float complexity_score;            
        float factual_score;               
        float creative_score;              
        int query_length;                
        bool contains_question_words;      
        bool contains_emergency_words;     
        bool contains_technical_words;   
    };

    struct QueryClassification
    {
        enum QueryType
        {
            FACTUAL_QUERY,   
            COMPLEX_QUERY,   
            CREATIVE_QUERY,  
            EMERGENCY_QUERY, 
            UNKNOWN_QUERY   
        };

        QueryType query_type;
        float confidence;
        std::string reasoning;
        bool requires_immediate_response;
    };
```

**向量搜索实现** - Python端的高效实现：

```python:88:112:Edge_LLM_RAG_Voice/automotive_edge_rag/python/vehicle_vector_search.py
    def search(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        if self.model is None:
            self.load_model(self.model_path)
        
        query_embedding = self.model.encode([query])
        
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                result = {
                    'id': idx,
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(similarity),
                    'section': self.metadata[idx]['section'],
                    'subsection': self.metadata[idx]['subsection']
                }
                results.append(result)
        
        return results
```

### 3. C++与Python混合编程架构

项目采用**C++主框架 + Python AI模块**的设计：

- C++ 负责系统控制、通信、流程管理
- Python 负责向量检索、模型推理
- 通过pybind11实现无缝集成

这种设计兼顾了**性能**和**AI生态支持**。

### 4. TTS语音合成系统

项目包含完整的TTS实现，从目录结构看：

- 使用Eigen数学库
- 包含文本规范化（Text Normalization）
- 支持中英文混合
- 采用VITs风格的声学模型

### 5. 基础设施层

**StackFlow框架** - 提供事件驱动的工作流管理：

```cpp:28:84:Edge-LLM-Infra/infra-controller/include/StackFlow.h
class StackFlow {
public:
    typedef enum {
        EVENT_NONE = 0,
        EVENT_SETUP,
        EVENT_EXIT,
        EVENT_PAUSE,
        EVENT_TASKINFO,
    } LOCAL_EVENT;

    std::string unit_name_;
    std::string request_id_;
    std::string out_zmq_url_;

    std::atomic<bool> exit_flage_;
    std::atomic<int> status_;

    // 线程安全的事件队列，存储事件类型
    eventpp::EventQueue<int, void(const std::shared_ptr<void> &)> event_queue_;
    std::unique_ptr<std::thread> even_loop_thread_;

    std::unique_ptr<pzmq> rpc_ctx_;

    std::unordered_map<int, std::shared_ptr<llm_channel_obj>> llm_task_channel_;

    StackFlow(const std::string &unit_name);
    void even_loop();
    void _none_event(const std::shared_ptr<void> &arg);

    template <typename T>
    std::shared_ptr<llm_channel_obj> get_channel(T workid)
    {
        int _work_id_num;
        if constexpr (std::is_same<T, int>::value) {
            _work_id_num = workid;
        } else if constexpr (std::is_same<T, std::string>::value) {
            _work_id_num = sample_get_work_id_num(workid);
        } else {
            return nullptr;
        }
        return llm_task_channel_.at(_work_id_num);
    }

    std::string _rpc_setup(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data);
    void _setup(const std::shared_ptr<void> &arg)
    {
        std::shared_ptr<pzmq_data> originalPtr = std::static_pointer_cast<pzmq_data>(arg);
        // data->get_param(0), data->get_param(1)
        std::string zmq_url = originalPtr->get_param(0);
        std::string data    = originalPtr->get_param(1);

        request_id_  = sample_json_str_get(data, "request_id");
        out_zmq_url_ = zmq_url;
        if (status_.load()) setup(zmq_url, data);
    };
    virtual int setup(const std::string &zmq_url, const std::string &raw);
    virtual int setup(const std::string &work_id, const std::string &object, const std::string &data);
```

## 技术亮点总结

1. **全离线部署** - 不依赖云端，适合车载等离线场景
2. **智能响应策略** - 根据查询类型自适应选择最优方案
3. **模块化设计** - 松耦合架构便于维护和扩展
4. **性能优化** - 缓存机制、预加载、向量索引
5. **实时性保证** - 流式ASR、双缓冲TTS队列
6. **工程化完善** - CMake构建、多语言混合、完整的数据处理流程

## 适用场景

- 自动驾驶/车载智能座舱
- 嵌入式AI设备
- 边缘计算场景
- 对隐私和离线能力有要求的应用

## 存在的问题和改进建议

从代码分析来看，系统整体架构合理，但有几点可以考虑：

1. **错误处理** - 部分地方可以增强异常处理
2. **日志系统** - 可以统一日志框架
3. **配置管理** - 硬编码的端口和路径可以配置化
4. **资源管理** - Python解释器和模型的生命周期管理可以优化

这是一个非常完整和实用的端侧AI系统项目，展现了在边缘设备上实现复杂AI能力的工程实践。是否需要我针对某个具体模块进行更深入的分析？

```plaintext
#pragma once

#include <Python.h>
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <memory>
#include <filesystem>

#include <unordered_map>
#include <atomic>
#include "query_classifier.h"
#include "ZmqServer.h"
#include "ZmqClient.h"

namespace fs = std::filesystem;

namespace py = pybind11;

namespace edge_llm_rag
{

    class EdgeLLMRAGSystem
    {
    public:
        explicit EdgeLLMRAGSystem();
        ~EdgeLLMRAGSystem();

        bool initialize();

        std::string process_query(const std::string &query,
                                  const std::string &user_id = "",
                                  const std::string &context = "");

        QueryClassification classify_query(const std::string &query);

        std::string rag_only_response(const std::string &query, bool preload = false);

        std::string llm_only_response(const std::string &query);

        std::string hybrid_response(const std::string &query);

        bool cleanup_cache();

    private:
        bool is_initialized_;

        py::object searcher;
        py::scoped_interpreter guard{};

        zmq_component::ZmqClient tts_client_{"tcp://localhost:7777"};
        zmq_component::ZmqClient llm_client_{"tcp://localhost:8899"};

        std::unique_ptr<QueryClassifier>
            query_classifier_;

        std::unordered_map<std::string, std::string> query_cache_;

        bool add_to_cache(const std::string &query, const std::string &response);
        std::string get_from_cache(const std::string &query);
        bool is_cache_valid(const std::string &query);

        void rag_message_worker(const std::string &rag_text);
        bool preload_common_queries();
    };

} // namespace edge_llm_rag
```

```plaintext
        switch (classification.query_type)
        {
        case QueryClassification::EMERGENCY_QUERY:
            std::cout << "===============================" << std::endl;
            std::cout << "紧急查询 detected, using RAG only response." << std::endl;
            std::cout << "===============================" << std::endl;
            response = rag_only_response(query);
            break;
        case QueryClassification::FACTUAL_QUERY:
            std::cout << "===============================" << std::endl;
            std::cout << "事实性查询 detected, using RAG only response." << std::endl;
            std::cout << "===============================" << std::endl;
            response = rag_only_response(query);
            break;
        case QueryClassification::COMPLEX_QUERY:
            std::cout << "===============================" << std::endl;
            std::cout << "复杂查询 detected, using hybrid response." << std::endl;
            std::cout << "===============================" << std::endl;

            response = hybrid_response(query);
            break;
        case QueryClassification::CREATIVE_QUERY:
            std::cout << "===============================" << std::endl;
            std::cout << "创意查询 detected, using LLM only response." << std::endl;
            std::cout << "===============================" << std::endl;
            response = llm_only_response(query);
            break;
        default:
            std::cout << "===============================" << std::endl;
            std::cout << "未知查询类型, using adaptive response." << std::endl;
            std::cout << "===============================" << std::endl;
            response = hybrid_response(query);
        }
```

```plaintext
    struct QueryFeatures
    {
        std::vector<std::string> keywords;
        float urgency_score;               
        float complexity_score;            
        float factual_score;               
        float creative_score;              
        int query_length;                
        bool contains_question_words;      
        bool contains_emergency_words;     
        bool contains_technical_words;   
    };

    struct QueryClassification
    {
        enum QueryType
        {
            FACTUAL_QUERY,   
            COMPLEX_QUERY,   
            CREATIVE_QUERY,  
            EMERGENCY_QUERY, 
            UNKNOWN_QUERY   
        };

        QueryType query_type;
        float confidence;
        std::string reasoning;
        bool requires_immediate_response;
    };
```

```plaintext
    def search(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        if self.model is None:
            self.load_model(self.model_path)
        
        query_embedding = self.model.encode([query])
        
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                result = {
                    'id': idx,
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(similarity),
                    'section': self.metadata[idx]['section'],
                    'subsection': self.metadata[idx]['subsection']
                }
                results.append(result)
        
        return results
```

```plaintext
class StackFlow {
public:
    typedef enum {
        EVENT_NONE = 0,
        EVENT_SETUP,
        EVENT_EXIT,
        EVENT_PAUSE,
        EVENT_TASKINFO,
    } LOCAL_EVENT;

    std::string unit_name_;
    std::string request_id_;
    std::string out_zmq_url_;

    std::atomic<bool> exit_flage_;
    std::atomic<int> status_;

    // 线程安全的事件队列，存储事件类型
    eventpp::EventQueue<int, void(const std::shared_ptr<void> &)> event_queue_;
    std::unique_ptr<std::thread> even_loop_thread_;

    std::unique_ptr<pzmq> rpc_ctx_;

    std::unordered_map<int, std::shared_ptr<llm_channel_obj>> llm_task_channel_;

    StackFlow(const std::string &unit_name);
    void even_loop();
    void _none_event(const std::shared_ptr<void> &arg);

    template <typename T>
    std::shared_ptr<llm_channel_obj> get_channel(T workid)
    {
        int _work_id_num;
        if constexpr (std::is_same<T, int>::value) {
            _work_id_num = workid;
        } else if constexpr (std::is_same<T, std::string>::value) {
            _work_id_num = sample_get_work_id_num(workid);
        } else {
            return nullptr;
        }
        return llm_task_channel_.at(_work_id_num);
    }

    std::string _rpc_setup(pzmq *_pzmq, const std::shared_ptr<pzmq_data> &data);
    void _setup(const std::shared_ptr<void> &arg)
    {
        std::shared_ptr<pzmq_data> originalPtr = std::static_pointer_cast<pzmq_data>(arg);
        // data->get_param(0), data->get_param(1)
        std::string zmq_url = originalPtr->get_param(0);
        std::string data    = originalPtr->get_param(1);

        request_id_  = sample_json_str_get(data, "request_id");
        out_zmq_url_ = zmq_url;
        if (status_.load()) setup(zmq_url, data);
    };
    virtual int setup(const std::string &zmq_url, const std::string &raw);
    virtual int setup(const std::string &work_id, const std::string &object, const std::string &data);
```

---

**User**

分析三个系统的区别和联系，及各自的功能

---

**Cursor**

[MODE: RESEARCH]

让我深入分析这三个子系统的具体差异和关联关系。

非常好，现在我已经深入分析了代码和架构，让我为您详细解析三个系统的区别、联系和功能定位。

## 三大系统的整体定位

这三个系统构成了一个**分层递进**的技术架构体系：

```
┌──────────────────────────────────────────────────┐
│        Edge_LLM_RAG_Voice (应用层)                │
│    完整的RAG知识库 + 语音交互应用                  │
└─────────────────┬────────────────────────────────┘
                  │ 基于
┌─────────────────▼────────────────────────────────┐
│        LLM_Voice_Flow (流水线层)                  │
│    基础的语音交互流水线系统                        │
└─────────────────┬────────────────────────────────┘
                  │ 依赖
┌─────────────────▼────────────────────────────────┐
│        Edge-LLM-Infra (基础设施层)                │
│    分布式通信框架和基础组件                        │
└──────────────────────────────────────────────────┘
```

---

## 系统一：Edge-LLM-Infra（基础设施层）

### 功能定位
**分布式边缘计算的通信基础设施框架**

### 核心功能

#### 1. **高级ZeroMQ封装 (pzmq)**
提供比原生ZeroMQ更易用的C++接口：

```cpp:56:76:Edge-LLM-Infra/hybrid-comm/include/pzmq.hpp
       pzmq(const std::string &server)
            : zmq_ctx_(NULL), zmq_socket_(NULL), rpc_server_(server), flage_(true), timeout_(3000)
        {
            if (server.find("://") != std::string::npos)
            {
                rpc_url_head_.clear();
            }
        }

        // 具体通信模式创建
        pzmq(const std::string &url, int mode, const msg_callback_fun &raw_call = nullptr)
            : zmq_ctx_(NULL), zmq_socket_(NULL), mode_(mode), flage_(true), timeout_(3000)
        {
            if ((url[0] != 'i') && (url[1] != 'p'))
            {
                rpc_url_head_.clear();
            }
            if (mode_ != ZMQ_RPC_FUN)
                creat(url, raw_call);
        }
```

**支持的通信模式：**
- PUB/SUB（发布订阅）
- PUSH/PULL（管道模式）
- REQ/REP（请求响应）
- **自定义RPC模式**（基于REQ/REP扩展）

#### 2. **StackFlow工作流引擎**
提供事件驱动的任务管理框架：

```cpp:28:56:Edge-LLM-Infra/infra-controller/include/StackFlow.h
class StackFlow {
public:
    typedef enum {
        EVENT_NONE = 0,
        EVENT_SETUP,
        EVENT_EXIT,
        EVENT_PAUSE,
        EVENT_TASKINFO,
    } LOCAL_EVENT;

    std::string unit_name_;
    std::string request_id_;
    std::string out_zmq_url_;

    std::atomic<bool> exit_flage_;
    std::atomic<int> status_;

    // 线程安全的事件队列，存储事件类型
    eventpp::EventQueue<int, void(const std::shared_ptr<void> &)> event_queue_;
    std::unique_ptr<std::thread> even_loop_thread_;

    std::unique_ptr<pzmq> rpc_ctx_;

    std::unordered_map<int, std::shared_ptr<llm_channel_obj>> llm_task_channel_;

    StackFlow(const std::string &unit_name);
    void even_loop();
    void _none_event(const std::shared_ptr<void> &arg);
```

**功能特性：**
- 线程安全的事件队列
- 支持任务setup/exit/pause等生命周期管理
- 多任务通道管理
- 异步事件处理

#### 3. **单元管理器（UnitManager）**
负责管理分布式节点：

```json:1:8:Edge-LLM-Infra/unit-manager/master_config.json
{
    "config_tcp_server": 10001,
    "config_zmq_min_port": 5010,
    "config_zmq_max_port": 5555,
    "config_zmq_s_format": "ipc:///tmp/llm/%i.sock",
    "config_zmq_c_format": "ipc:///tmp/llm/%i.sock"

}
```

**管理功能：**
- 动态端口分配
- IPC通道创建与回收
- 节点注册与发现
- 会话管理

#### 4. **示例应用**

**RPC服务端示例：**

```cpp:8:18:Edge-LLM-Infra/sample/rpc_server.cc
std::string fun1_(pzmq* self, const std::shared_ptr<pzmq_data>& msg) {
    std::string raw_msg = msg->string();
    std::cout << "fun1 received: " << raw_msg << std::endl;
    return "hello fun1";
}

std::string fun2_(pzmq* self, const std::shared_ptr<pzmq_data>& msg) {
    std::string raw_msg = msg->string();
    std::cout << "fun2 received: " << raw_msg << std::endl;
    return "hello fun2";
}
```

**发布订阅示例：**

```cpp:12:26:Edge-LLM-Infra/sample/pub.cc
        // 1. 创建PUB套接字
        pzmq zpush_("ipc:///tmp/5001.socket", ZMQ_PUB);

        // 2. 等待连接建立（PUB-SUB模式需要）
        std::cout << "Publisher started. Waiting for subscribers..." << std::endl;
        sleep(1);  // 重要：给订阅者时间连接

        // 3. 发送消息
        int count = 0;
        while (true) {
            std::string msg = "Message " + std::to_string(++count);
            zpush_.send_data(msg);
            std::cout << "Sent: " << msg << std::endl;
            sleep(1);  // 每秒发送一条
        }
```

### 适用场景
- 需要分布式通信的边缘计算应用
- 多进程/多节点协同工作场景
- 复杂事件驱动的系统架构

---

## 系统二：LLM_Voice_Flow（流水线层）

### 功能定位
**基础的端到端语音交互流水线，专注于ASR→LLM→TTS核心流程**

### 系统架构

```
┌──────────┐      ┌──────────┐      ┌──────────┐
│   ASR    │─────▶│   LLM    │─────▶│   TTS    │
│ (sherpa) │ ZMQ  │(DeepSeek)│ ZMQ  │(双缓冲)  │
└──────────┘      └──────────┘      └──────────┘
     │                  │                  │
     │                  │                  │
  流式识别         RK NPU推理          音频播放
```

### 核心模块

#### 1. **ASR模块（sherpa-onnx）**
- 使用sherpa-onnx开源框架
- 支持流式语音识别
- 支持中英文双语模型
- 模型路径：`voice/models/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16/`

#### 2. **LLM模块（RKNN-LLM）**
基于RK芯片的专用LLM推理框架：

```markdown:1:41:Edge_LLM_RAG_Voice/llm/rknn-llm/README.md
# Description

  RKLLM software stack can help users to quickly deploy AI models to Rockchip chips. The overall framework is as follows:
    <center class="half">
        <div style="background-color:#ffffff;">
        <img src="res/framework.jpg" title="RKLLM"/>
    </center>

  In order to use RKNPU, users need to first run the RKLLM-Toolkit tool on the computer, convert the trained model into an RKLLM format model, and then inference on the development board using the RKLLM C API.

- RKLLM-Toolkit is a software development kit for users to perform model conversionand quantization on PC.

- RKLLM Runtime provides C/C++ programming interfaces for Rockchip NPU platform to help users deploy RKLLM models and accelerate the implementation of LLM applications.

- RKNPU kernel driver is responsible for interacting with NPU hardware. It has been open source and can be found in the Rockchip kernel code.

# Support Platform

- RK3588 Series
- RK3576 Series
- RK3562 Series

# Support Models

- [x] [LLAMA models](https://huggingface.co/meta-llama) 
- [x] [TinyLLAMA models](https://huggingface.co/TinyLlama) 
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b/tree/103caa40027ebfd8450289ca2f278eac4ff26405)
- [x] [Gemma2](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)
- [x] [Gemma3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)
- [x] [InternLM2 models](https://huggingface.co/collections/internlm/internlm2-65b0ce04970888799707893c)
- [x] [MiniCPM models](https://huggingface.co/collections/openbmb/minicpm-65d48bf958302b9fd25b698f)
- [x] [TeleChat models](https://huggingface.co/Tele-AI)
- [x] [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [x] [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
- [x] [DeepSeek-R1-Distill](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d)
- [x] [Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)
- [x] [InternVL2-1B](https://huggingface.co/OpenGVLab/InternVL2-1B)
- [x] [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
```

**性能特点：**
- 支持W4A16、W8A8等量化方案
- 针对RK NPU优化
- 支持最长16K上下文
- DeepSeek-R1模型示例位于`llm/rknn-llm/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/`

#### 3. **TTS模块（双缓冲队列设计）**

```cpp:61:104:LLM_Voice_Flow/tts/tts_server/src/main.cpp
int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    try {
    
        TTSModel model(argv[1]);
        AudioPlayer player;
        DoubleMessageQueue queue;

        std::thread synthesis_thread(synthesis_worker, std::ref(queue), std::ref(model));
        std::thread playback_thread(playback_worker, std::ref(queue), std::ref(player));

        while (true) {
            if (first_msg) {
                std::string req = status_server.receive();
                std::cout << "[voice -> tts] received: " << req << std::endl;
            }
            first_msg = false;

            std::string text = server.receive();
            server.send("Echo: received");
            std::cout << "[llm -> tts] received: " << text << std::endl;

            if (!text.empty() && text.find("<think>") == std::string::npos) {
              
    
                queue.push_text(text);
            }
        }

        // 清理
        queue.stop();
        synthesis_thread.join();
        playback_thread.join();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

**设计特点：**
- 双缓冲队列减少延迟
- 合成线程和播放线程分离
- 自动过滤思考过程（`<think>`标签）
- ZMQ接收LLM输出

### 系统特点
- **纯净的流水线设计**，无额外业务逻辑
- **模块独立运行**，通过ZMQ通信
- **快速响应**，目标4秒内完成闭环
- **易于扩展**，可替换任意模块

### 适用场景
- 简单的语音助手
- 语音对话系统
- 需要定制化LLM的场景

---

## 系统三：Edge_LLM_RAG_Voice（应用层）

### 功能定位
**企业级车载座舱智能问答系统，集成RAG知识库的完整应用**

### 系统架构

```
┌─────────────────────────────────────────────┐
│         用户语音输入（ASR）                  │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│      智能查询分类器 (QueryClassifier)        │
│  ├─ 紧急查询 (EMERGENCY_QUERY)               │
│  ├─ 事实性查询 (FACTUAL_QUERY)               │
│  ├─ 复杂查询 (COMPLEX_QUERY)                 │
│  └─ 创意查询 (CREATIVE_QUERY)                │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼────┐    ┌──────▼────────┐
│ RAG Only  │    │  LLM Only     │    
│ 快速响应  │    │  创意回答     │    
└───────────┘    └───────────────┘
       │                │
       └───────┬────────┘
               │
┌──────────────▼──────────────────────────────┐
│           混合响应 (Hybrid)                  │
│        RAG检索 + LLM增强理解                 │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│         TTS合成 + 语音输出                   │
└─────────────────────────────────────────────┘
```

### 核心创新：多级响应策略

#### 1. **智能查询分类**

```cpp:11:39:Edge_LLM_RAG_Voice/automotive_edge_rag/cpp/query_classifier.h
    struct QueryFeatures
    {
        std::vector<std::string> keywords;
        float urgency_score;               
        float complexity_score;            
        float factual_score;               
        float creative_score;              
        int query_length;                
        bool contains_question_words;      
        bool contains_emergency_words;     
        bool contains_technical_words;   
    };

    struct QueryClassification
    {
        enum QueryType
        {
            FACTUAL_QUERY,   
            COMPLEX_QUERY,   
            CREATIVE_QUERY,  
            EMERGENCY_QUERY, 
            UNKNOWN_QUERY   
        };

        QueryType query_type;
        float confidence;
        std::string reasoning;
        bool requires_immediate_response;
    };
```

**分类依据：**
- 关键词分析
- 紧急程度评分
- 复杂度评分
- 问句特征检测

#### 2. **RAG知识库系统**

**向量检索实现：**

```python:88:112:Edge_LLM_RAG_Voice/automotive_edge_rag/python/vehicle_vector_search.py
    def search(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        if self.model is None:
            self.load_model(self.model_path)
        
        query_embedding = self.model.encode([query])
        
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                result = {
                    'id': idx,
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(similarity),
                    'section': self.metadata[idx]['section'],
                    'subsection': self.metadata[idx]['subsection']
                }
                results.append(result)
        
        return results
```

**知识库特性：**
- 基于SentenceTransformer的向量化
- 余弦相似度检索
- 章节结构化元数据
- 相似度阈值过滤

#### 3. **三种响应模式**

**模式一：RAG Only（最快）**

```cpp:194:233:Edge_LLM_RAG_Voice/automotive_edge_rag/cpp/edge_llm_rag_system.cpp
    std::string EdgeLLMRAGSystem::rag_only_response(const std::string &query, bool preload)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        py::object results = searcher.attr("search")(query, 1, 0.5);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "\nQuery: '" << query << "'\n";
        std::cout
            << "elapsed: " << std::fixed << std::setprecision(2) << ms << " ms\n";

        if (py::len(results) == 0)
        {
            std::cout << "  No results" << std::endl;
            return "No results !!!";
        }

        std::string answer;
        for (const auto &item : results)
        {
            double sim = item["similarity"].cast<double>();
            std::string text = item["text"].cast<std::string>();
            answer = item["text"].cast<std::string>();
            std::string section = item["section"].cast<std::string>();
            std::string subsection = item["subsection"].cast<std::string>();
            std::cout << "  sim=" << std::fixed << std::setprecision(4) << sim
                      << ", section=" << section
                      << (subsection.empty() ? "" : ("/" + subsection))
                      << ", text=" << text.substr(0, 100) << "...\n";
        }

        if (!preload)
        {
            rag_message_worker(answer);
        }

        // std::string llm_query = query + "<rag>" + answer;
        // std::string llm_part = llm_only_response(llm_query);
        return answer;
    }
```

**适用于：** 紧急查询、事实性查询（如"发动机故障灯亮了怎么办"）

**模式二：LLM Only（灵活）**

```cpp:235:240:Edge_LLM_RAG_Voice/automotive_edge_rag/cpp/edge_llm_rag_system.cpp
    std::string EdgeLLMRAGSystem::llm_only_response(const std::string &query)
    {
        auto response = llm_client_.request(query);
        std::cout << "[tts -> RAG] received: " << response << std::endl;
        return response;
    }
```

**适用于：** 创意查询（如"给我讲个笑话"）

**模式三：Hybrid（精准）**

```cpp:242:253:Edge_LLM_RAG_Voice/automotive_edge_rag/cpp/edge_llm_rag_system.cpp
    std::string EdgeLLMRAGSystem::hybrid_response(const std::string &query)
    {
        // 结合RAG和LLM的响应
        std::string rag_part = rag_only_response(query,true);
        if (rag_part.find("No results") != std::string::npos){
            return llm_only_response(query);
        }
        std::string llm_query = query + "<rag>" + rag_part;
        std::string llm_part = llm_only_response(llm_query);

        return llm_part;
    }
```

**适用于：** 复杂查询（如"为什么我的车油耗突然增加了"）

#### 4. **智能TTS分段**

```cpp:138:192:Edge_LLM_RAG_Voice/automotive_edge_rag/cpp/edge_llm_rag_system.cpp
    void EdgeLLMRAGSystem::rag_message_worker(const std::string &rag_text)
    {
        static const std::wregex wide_delimiter(
            L"([。！？；：\n]|\\?\\s|\\!\\s|\\；|\\，|\\、|\\|)");
        const std::wstring END_MARKER = L"END";

        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

        std::wstring wide_text = converter.from_bytes(rag_text) + END_MARKER;

        std::wsregex_iterator it(wide_text.begin(), wide_text.end(), wide_delimiter);
        std::wsregex_iterator end;

        int skip_counter = 0;
        size_t last_pos = 0;
        while (it != end && skip_counter < 2)
        {
            last_pos = it->position() + it->length();
            ++it;
            ++skip_counter;
        }

        while (it != end)
        {
            size_t seg_start = last_pos;
            size_t seg_end = it->position();
            last_pos = seg_end + it->length();

            std::wstring wide_segment = wide_text.substr(seg_start, seg_end - seg_start);

            wide_segment.erase(0, wide_segment.find_first_not_of(L" \t\n\r"));
            wide_segment.erase(wide_segment.find_last_not_of(L" \t\n\r") + 1);

            if (!wide_segment.empty())
            {
                // 转换回UTF-8
                auto response1 = tts_client_.request(converter.to_bytes(wide_segment));
                std::cout << "[tts -> RAG] received: " << response1 << std::endl;
                // queue.push_text(converter.to_bytes(wide_segment));
            }
            ++it;
        }

        // 处理剩余内容
        if (last_pos < wide_text.length())
        {
            std::wstring last_segment = wide_text.substr(last_pos);
            if (!last_segment.empty())
            {
                auto response1 = tts_client_.request(converter.to_bytes(last_segment));
                std::cout << "[tts -> RAG] received: " << response1 << std::endl;
                // queue.push_text(converter.to_bytes(last_segment));
            }
        }
    }
```

**优化特性：**
- 按标点符号智能分段
- 跳过前两句（可能是寒暄）
- Unicode处理确保中文正确
- 实时流式发送TTS

#### 5. **缓存机制**

```cpp:255:279:Edge_LLM_RAG_Voice/automotive_edge_rag/cpp/edge_llm_rag_system.cpp
    bool EdgeLLMRAGSystem::add_to_cache(const std::string &query, const std::string &response)
    {
        if (query_cache_.size() >= 100)
        { // 限制缓存大小
            query_cache_.clear();
        }

        query_cache_[query] = response;
        return true;
    }

    std::string EdgeLLMRAGSystem::get_from_cache(const std::string &query)
    {
        auto it = query_cache_.find(query);
        if (it != query_cache_.end())
        {
            return it->second;
        }
        return "";
    }

    bool EdgeLLMRAGSystem::is_cache_valid(const std::string &query)
    {
        return query_cache_.find(query) != query_cache_.end();
    }
```

### 系统特点
- **业务场景明确** - 专为车载座舱设计
- **知识库驱动** - RAG检索车辆手册
- **智能路由** - 根据查询类型选择最优策略
- **性能优化** - 缓存、预加载、分段处理

### 适用场景
- 车载智能问答系统
- 需要结构化知识库的场景
- 对响应速度和准确性都有要求的应用

---

## 三大系统的关系总结

### 依赖关系

```
Edge_LLM_RAG_Voice
    ├─ 使用 LLM_Voice_Flow 的 TTS 模块
    ├─ 使用 LLM_Voice_Flow 的 LLM 推理能力
    ├─ 使用 Edge-LLM-Infra 的 ZmqClient/ZmqServer
    └─ 扩展 LLM_Voice_Flow，增加 RAG 能力

LLM_Voice_Flow
    ├─ 使用 Edge-LLM-Infra 的 ZeroMQ 封装
    └─ 提供基础的语音交互能力

Edge-LLM-Infra
    └─ 独立的基础设施，被其他系统依赖
```

### 演进路径

```
第一步：Edge-LLM-Infra
        搭建通信基础设施

         ↓

第二步：LLM_Voice_Flow
        实现基础语音交互流水线

         ↓

第三步：Edge_LLM_RAG_Voice
        在流水线基础上增加RAG知识库
```

### 功能对比表

| 功能特性 | Edge-LLM-Infra | LLM_Voice_Flow | Edge_LLM_RAG_Voice |
|---------|----------------|----------------|-------------------|
| **通信框架** | ✅ 核心提供 | ✅ 使用 | ✅ 使用 |
| **ASR模块** | ❌ | ✅ | ✅（共享） |
| **LLM推理** | ❌ | ✅ | ✅（共享） |
| **TTS合成** | ❌ | ✅ | ✅（共享） |
| **RAG检索** | ❌ | ❌ | ✅ 核心创新 |
| **查询分类** | ❌ | ❌ | ✅ 独有 |
| **响应策略** | ❌ | 单一流程 | ✅ 多级策略 |
| **知识库** | ❌ | ❌ | ✅ 车辆手册 |
| **缓存机制** | ❌ | ❌ | ✅ |

### 选择建议

**选择 Edge-LLM-Infra 如果你需要：**
- 构建分布式边缘计算系统
- 灵活的进程间通信框架
- 自定义工作流引擎

**选择 LLM_Voice_Flow 如果你需要：**
- 快速搭建语音助手原型
- 简单的语音对话功能
- 无需知识库的通用对话

**选择 Edge_LLM_RAG_Voice 如果你需要：**
- 车载智能问答系统
- 基于结构化知识的准确回答
- 智能的响应策略路由
- 企业级的性能和可靠性

这三个系统构成了一个完整的技术栈，从底层通信到业务应用层层递进，既可以独立使用，也可以组合使用，非常适合嵌入式AI和边缘计算场景的开发需求。

```plaintext
┌──────────────────────────────────────────────────┐
│        Edge_LLM_RAG_Voice (应用层)                │
│    完整的RAG知识库 + 语音交互应用                  │
└─────────────────┬────────────────────────────────┘
                  │ 基于
┌─────────────────▼────────────────────────────────┐
│        LLM_Voice_Flow (流水线层)                  │
│    基础的语音交互流水线系统                        │
└─────────────────┬────────────────────────────────┘
                  │ 依赖
┌─────────────────▼────────────────────────────────┐
│        Edge-LLM-Infra (基础设施层)                │
│    分布式通信框架和基础组件                        │
└──────────────────────────────────────────────────┘
```

```plaintext
       pzmq(const std::string &server)
            : zmq_ctx_(NULL), zmq_socket_(NULL), rpc_server_(server), flage_(true), timeout_(3000)
        {
            if (server.find("://") != std::string::npos)
            {
                rpc_url_head_.clear();
            }
        }

        // 具体通信模式创建
        pzmq(const std::string &url, int mode, const msg_callback_fun &raw_call = nullptr)
            : zmq_ctx_(NULL), zmq_socket_(NULL), mode_(mode), flage_(true), timeout_(3000)
        {
            if ((url[0] != 'i') && (url[1] != 'p'))
            {
                rpc_url_head_.clear();
            }
            if (mode_ != ZMQ_RPC_FUN)
                creat(url, raw_call);
        }
```

```plaintext
class StackFlow {
public:
    typedef enum {
        EVENT_NONE = 0,
        EVENT_SETUP,
        EVENT_EXIT,
        EVENT_PAUSE,
        EVENT_TASKINFO,
    } LOCAL_EVENT;

    std::string unit_name_;
    std::string request_id_;
    std::string out_zmq_url_;

    std::atomic<bool> exit_flage_;
    std::atomic<int> status_;

    // 线程安全的事件队列，存储事件类型
    eventpp::EventQueue<int, void(const std::shared_ptr<void> &)> event_queue_;
    std::unique_ptr<std::thread> even_loop_thread_;

    std::unique_ptr<pzmq> rpc_ctx_;

    std::unordered_map<int, std::shared_ptr<llm_channel_obj>> llm_task_channel_;

    StackFlow(const std::string &unit_name);
    void even_loop();
    void _none_event(const std::shared_ptr<void> &arg);
```

```plaintext
{
    "config_tcp_server": 10001,
    "config_zmq_min_port": 5010,
    "config_zmq_max_port": 5555,
    "config_zmq_s_format": "ipc:///tmp/llm/%i.sock",
    "config_zmq_c_format": "ipc:///tmp/llm/%i.sock"

}
```

```plaintext
std::string fun1_(pzmq* self, const std::shared_ptr<pzmq_data>& msg) {
    std::string raw_msg = msg->string();
    std::cout << "fun1 received: " << raw_msg << std::endl;
    return "hello fun1";
}

std::string fun2_(pzmq* self, const std::shared_ptr<pzmq_data>& msg) {
    std::string raw_msg = msg->string();
    std::cout << "fun2 received: " << raw_msg << std::endl;
    return "hello fun2";
}
```

```plaintext
        // 1. 创建PUB套接字
        pzmq zpush_("ipc:///tmp/5001.socket", ZMQ_PUB);

        // 2. 等待连接建立（PUB-SUB模式需要）
        std::cout << "Publisher started. Waiting for subscribers..." << std::endl;
        sleep(1);  // 重要：给订阅者时间连接

        // 3. 发送消息
        int count = 0;
        while (true) {
            std::string msg = "Message " + std::to_string(++count);
            zpush_.send_data(msg);
            std::cout << "Sent: " << msg << std::endl;
            sleep(1);  // 每秒发送一条
        }
```

```plaintext
┌──────────┐      ┌──────────┐      ┌──────────┐
│   ASR    │─────▶│   LLM    │─────▶│   TTS    │
│ (sherpa) │ ZMQ  │(DeepSeek)│ ZMQ  │(双缓冲)  │
└──────────┘      └──────────┘      └──────────┘
     │                  │                  │
     │                  │                  │
  流式识别         RK NPU推理          音频播放
```

```plaintext
# Description

  RKLLM software stack can help users to quickly deploy AI models to Rockchip chips. The overall framework is as follows:
    <center class="half">
        <div style="background-color:#ffffff;">
        <img src="res/framework.jpg" title="RKLLM"/>
    </center>

  In order to use RKNPU, users need to first run the RKLLM-Toolkit tool on the computer, convert the trained model into an RKLLM format model, and then inference on the development board using the RKLLM C API.

- RKLLM-Toolkit is a software development kit for users to perform model conversionand quantization on PC.

- RKLLM Runtime provides C/C++ programming interfaces for Rockchip NPU platform to help users deploy RKLLM models and accelerate the implementation of LLM applications.

- RKNPU kernel driver is responsible for interacting with NPU hardware. It has been open source and can be found in the Rockchip kernel code.

# Support Platform

- RK3588 Series
- RK3576 Series
- RK3562 Series

# Support Models

- [x] [LLAMA models](https://huggingface.co/meta-llama) 
- [x] [TinyLLAMA models](https://huggingface.co/TinyLlama) 
- [x] [Qwen models](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [Phi models](https://huggingface.co/models?search=microsoft/phi)
- [x] [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b/tree/103caa40027ebfd8450289ca2f278eac4ff26405)
- [x] [Gemma2](https://huggingface.co/collections/google/gemma-2-release-667d6600fd5220e7b967f315)
- [x] [Gemma3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d)
- [x] [InternLM2 models](https://huggingface.co/collections/internlm/internlm2-65b0ce04970888799707893c)
- [x] [MiniCPM models](https://huggingface.co/collections/openbmb/minicpm-65d48bf958302b9fd25b698f)
- [x] [TeleChat models](https://huggingface.co/Tele-AI)
- [x] [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
- [x] [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
- [x] [DeepSeek-R1-Distill](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d)
- [x] [Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)
- [x] [InternVL2-1B](https://huggingface.co/OpenGVLab/InternVL2-1B)
- [x] [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
```

```plaintext
int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    try {
    
        TTSModel model(argv[1]);
        AudioPlayer player;
        DoubleMessageQueue queue;

        std::thread synthesis_thread(synthesis_worker, std::ref(queue), std::ref(model));
        std::thread playback_thread(playback_worker, std::ref(queue), std::ref(player));

        while (true) {
            if (first_msg) {
                std::string req = status_server.receive();
                std::cout << "[voice -> tts] received: " << req << std::endl;
            }
            first_msg = false;

            std::string text = server.receive();
            server.send("Echo: received");
            std::cout << "[llm -> tts] received: " << text << std::endl;

            if (!text.empty() && text.find("<think>") == std::string::npos) {
              
    
                queue.push_text(text);
            }
        }

        // 清理
        queue.stop();
        synthesis_thread.join();
        playback_thread.join();
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

```plaintext
┌─────────────────────────────────────────────┐
│         用户语音输入（ASR）                  │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│      智能查询分类器 (QueryClassifier)        │
│  ├─ 紧急查询 (EMERGENCY_QUERY)               │
│  ├─ 事实性查询 (FACTUAL_QUERY)               │
│  ├─ 复杂查询 (COMPLEX_QUERY)                 │
│  └─ 创意查询 (CREATIVE_QUERY)                │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴────────┐
       │                │
┌──────▼────┐    ┌──────▼────────┐
│ RAG Only  │    │  LLM Only     │    
│ 快速响应  │    │  创意回答     │    
└───────────┘    └───────────────┘
       │                │
       └───────┬────────┘
               │
┌──────────────▼──────────────────────────────┐
│           混合响应 (Hybrid)                  │
│        RAG检索 + LLM增强理解                 │
└──────────────┬──────────────────────────────┘
               │
┌──────────────▼──────────────────────────────┐
│         TTS合成 + 语音输出                   │
└─────────────────────────────────────────────┘
```

```plaintext
    struct QueryFeatures
    {
        std::vector<std::string> keywords;
        float urgency_score;               
        float complexity_score;            
        float factual_score;               
        float creative_score;              
        int query_length;                
        bool contains_question_words;      
        bool contains_emergency_words;     
        bool contains_technical_words;   
    };

    struct QueryClassification
    {
        enum QueryType
        {
            FACTUAL_QUERY,   
            COMPLEX_QUERY,   
            CREATIVE_QUERY,  
            EMERGENCY_QUERY, 
            UNKNOWN_QUERY   
        };

        QueryType query_type;
        float confidence;
        std::string reasoning;
        bool requires_immediate_response;
    };
```

```plaintext
    def search(self, query: str, top_k: int = 5, threshold: float = 0.5) -> List[Dict[str, Any]]:
        if self.model is None:
            self.load_model(self.model_path)
        
        query_embedding = self.model.encode([query])
        
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                result = {
                    'id': idx,
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(similarity),
                    'section': self.metadata[idx]['section'],
                    'subsection': self.metadata[idx]['subsection']
                }
                results.append(result)
        
        return results
```

```plaintext
    std::string EdgeLLMRAGSystem::rag_only_response(const std::string &query, bool preload)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        py::object results = searcher.attr("search")(query, 1, 0.5);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "\nQuery: '" << query << "'\n";
        std::cout
            << "elapsed: " << std::fixed << std::setprecision(2) << ms << " ms\n";

        if (py::len(results) == 0)
        {
            std::cout << "  No results" << std::endl;
            return "No results !!!";
        }

        std::string answer;
        for (const auto &item : results)
        {
            double sim = item["similarity"].cast<double>();
            std::string text = item["text"].cast<std::string>();
            answer = item["text"].cast<std::string>();
            std::string section = item["section"].cast<std::string>();
            std::string subsection = item["subsection"].cast<std::string>();
            std::cout << "  sim=" << std::fixed << std::setprecision(4) << sim
                      << ", section=" << section
                      << (subsection.empty() ? "" : ("/" + subsection))
                      << ", text=" << text.substr(0, 100) << "...\n";
        }

        if (!preload)
        {
            rag_message_worker(answer);
        }

        // std::string llm_query = query + "<rag>" + answer;
        // std::string llm_part = llm_only_response(llm_query);
        return answer;
    }
```

```plaintext
    std::string EdgeLLMRAGSystem::llm_only_response(const std::string &query)
    {
        auto response = llm_client_.request(query);
        std::cout << "[tts -> RAG] received: " << response << std::endl;
        return response;
    }
```

```plaintext
    std::string EdgeLLMRAGSystem::hybrid_response(const std::string &query)
    {
        // 结合RAG和LLM的响应
        std::string rag_part = rag_only_response(query,true);
        if (rag_part.find("No results") != std::string::npos){
            return llm_only_response(query);
        }
        std::string llm_query = query + "<rag>" + rag_part;
        std::string llm_part = llm_only_response(llm_query);

        return llm_part;
    }
```

```plaintext
    void EdgeLLMRAGSystem::rag_message_worker(const std::string &rag_text)
    {
        static const std::wregex wide_delimiter(
            L"([。！？；：\n]|\\?\\s|\\!\\s|\\；|\\，|\\、|\\|)");
        const std::wstring END_MARKER = L"END";

        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

        std::wstring wide_text = converter.from_bytes(rag_text) + END_MARKER;

        std::wsregex_iterator it(wide_text.begin(), wide_text.end(), wide_delimiter);
        std::wsregex_iterator end;

        int skip_counter = 0;
        size_t last_pos = 0;
        while (it != end && skip_counter < 2)
        {
            last_pos = it->position() + it->length();
            ++it;
            ++skip_counter;
        }

        while (it != end)
        {
            size_t seg_start = last_pos;
            size_t seg_end = it->position();
            last_pos = seg_end + it->length();

            std::wstring wide_segment = wide_text.substr(seg_start, seg_end - seg_start);

            wide_segment.erase(0, wide_segment.find_first_not_of(L" \t\n\r"));
            wide_segment.erase(wide_segment.find_last_not_of(L" \t\n\r") + 1);

            if (!wide_segment.empty())
            {
                // 转换回UTF-8
                auto response1 = tts_client_.request(converter.to_bytes(wide_segment));
                std::cout << "[tts -> RAG] received: " << response1 << std::endl;
                // queue.push_text(converter.to_bytes(wide_segment));
            }
            ++it;
        }

        // 处理剩余内容
        if (last_pos < wide_text.length())
        {
            std::wstring last_segment = wide_text.substr(last_pos);
            if (!last_segment.empty())
            {
                auto response1 = tts_client_.request(converter.to_bytes(last_segment));
                std::cout << "[tts -> RAG] received: " << response1 << std::endl;
                // queue.push_text(converter.to_bytes(last_segment));
            }
        }
    }
```

```plaintext
    bool EdgeLLMRAGSystem::add_to_cache(const std::string &query, const std::string &response)
    {
        if (query_cache_.size() >= 100)
        { // 限制缓存大小
            query_cache_.clear();
        }

        query_cache_[query] = response;
        return true;
    }

    std::string EdgeLLMRAGSystem::get_from_cache(const std::string &query)
    {
        auto it = query_cache_.find(query);
        if (it != query_cache_.end())
        {
            return it->second;
        }
        return "";
    }

    bool EdgeLLMRAGSystem::is_cache_valid(const std::string &query)
    {
        return query_cache_.find(query) != query_cache_.end();
    }
```

```plaintext
Edge_LLM_RAG_Voice
    ├─ 使用 LLM_Voice_Flow 的 TTS 模块
    ├─ 使用 LLM_Voice_Flow 的 LLM 推理能力
    ├─ 使用 Edge-LLM-Infra 的 ZmqClient/ZmqServer
    └─ 扩展 LLM_Voice_Flow，增加 RAG 能力

LLM_Voice_Flow
    ├─ 使用 Edge-LLM-Infra 的 ZeroMQ 封装
    └─ 提供基础的语音交互能力

Edge-LLM-Infra
    └─ 独立的基础设施，被其他系统依赖
```

```plaintext
第一步：Edge-LLM-Infra
        搭建通信基础设施

         ↓

第二步：LLM_Voice_Flow
        实现基础语音交互流水线

         ↓

第三步：Edge_LLM_RAG_Voice
        在流水线基础上增加RAG知识库
```

---

**User**

@bash (1-18) 分析出错原因

---

**Cursor**

[MODE: RESEARCH]

让我分析这个Docker构建错误的原因。

```plaintext
ERROR: failed to build: failed to solve: ubuntu:20.04: failed to resolve source metadata for docker.io/library/ubuntu:20.04: failed to do request: Head "https://docker.anyhub.us.kg/v2/library/ubuntu/manifests/20.04?ns=docker.io": EOF
```

## 错误原因分析

根据终端输出，这是一个**Docker镜像拉取失败**的问题。具体分析如下：

### 核心错误信息

```bash
ERROR: failed to resolve source metadata for docker.io/library/ubuntu:20.04: 
failed to do request: Head "https://docker.anyhub.us.kg/v2/library/ubuntu/manifests/20.04?ns=docker.io": EOF
```

### 问题根源

1. **镜像源不可用**
   - Docker正在尝试从 `docker.anyhub.us.kg` 拉取镜像
   - 这个镜像代理地址返回 `EOF` (连接意外终止)
   - 该镜像源可能已失效或网络不通

2. **网络连接问题**
   - 即使使用了 `--network host` 参数
   - 镜像源本身无法访问或响应超时

3. **镜像源配置**
   - 您的Docker配置中使用了一个不稳定的镜像源

### 解决方案

#### 方案一：修改Docker镜像源配置（推荐）

编辑Docker配置文件：

```bash
sudo vim /etc/docker/daemon.json
```

添加或修改为以下内容（使用国内可用源）：

```json
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
```

重启Docker服务：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

#### 方案二：使用阿里云镜像加速器

1. 登录阿里云容器镜像服务：https://cr.console.aliyun.com/cn-hangzhou/instances/mirrors
2. 获取专属加速地址
3. 配置到 `/etc/docker/daemon.json`：

```json
{
  "registry-mirrors": ["https://你的专属地址.mirror.aliyuncs.com"]
}
```

#### 方案三：临时解决 - 预先拉取镜像

```bash
# 手动拉取基础镜像
docker pull ubuntu:20.04

# 或使用代理
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
docker pull ubuntu:20.04
```

#### 方案四：修改Dockerfile使用本地镜像

如果已有ubuntu:20.04镜像，确保本地存在：

```bash
docker images | grep ubuntu
```

### 验证镜像源是否可用

测试当前配置：

```bash
# 查看当前Docker配置
docker info | grep -A 10 "Registry Mirrors"

# 测试镜像拉取
docker pull hello-world
```

### 推荐操作步骤

```bash
# 1. 停止Docker
sudo systemctl stop docker

# 2. 备份现有配置
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak

# 3. 修改配置为中科大源
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://docker.mirrors.ustc.edu.cn"]
}
EOF

# 4. 重启Docker
sudo systemctl daemon-reload
sudo systemctl start docker

# 5. 重新构建
cd /home/chunyu/hard_disk_1/intelligent_voice/Edge-LLM-Infra/docker/build
docker build --network host -f base.dockerfile . -t llm:v1.0
```

**注意**：`docker.anyhub.us.kg` 这个源已经不稳定，建议更换为上述推荐的镜像源。

```shellscript
ERROR: failed to resolve source metadata for docker.io/library/ubuntu:20.04: 
failed to do request: Head "https://docker.anyhub.us.kg/v2/library/ubuntu/manifests/20.04?ns=docker.io": EOF
```

```shellscript
sudo vim /etc/docker/daemon.json
```

```json
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
```

```shellscript
sudo systemctl daemon-reload
sudo systemctl restart docker
```

```json
{
  "registry-mirrors": ["https://你的专属地址.mirror.aliyuncs.com"]
}
```

```shellscript
# 手动拉取基础镜像
docker pull ubuntu:20.04

# 或使用代理
export http_proxy=http://your-proxy:port
export https_proxy=http://your-proxy:port
docker pull ubuntu:20.04
```

```shellscript
docker images | grep ubuntu
```

```shellscript
# 查看当前Docker配置
docker info | grep -A 10 "Registry Mirrors"

# 测试镜像拉取
docker pull hello-world
```

```shellscript
# 1. 停止Docker
sudo systemctl stop docker

# 2. 备份现有配置
sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak

# 3. 修改配置为中科大源
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://docker.mirrors.ustc.edu.cn"]
}
EOF

# 4. 重启Docker
sudo systemctl daemon-reload
sudo systemctl start docker

# 5. 重新构建
cd /home/chunyu/hard_disk_1/intelligent_voice/Edge-LLM-Infra/docker/build
docker build --network host -f base.dockerfile . -t llm:v1.0
```

---

**User**

@Edge-LLM-Infra/unit-manager/src/main.cpp:68-72 解释该函数的作用

---

**Cursor**

[MODE: RESEARCH]

让我查看这段代码的完整上下文，以便更好地解释它的作用。

## 函数作用详解

这段代码的核心是**初始化一个自旋锁（Spinlock）**，用于保护共享数据结构的并发访问。

### 完整解析

```cpp:68:72:Edge-LLM-Infra/unit-manager/src/main.cpp
if (pthread_spin_init(&key_sql_lock, PTHREAD_PROCESS_PRIVATE) != 0)
{
    ALOGE("key_sql_lock init false");
    exit(1);
}
```

### 函数签名

```cpp
int pthread_spin_init(pthread_spinlock_t *lock, int pshared);
```

### 参数说明

**第一个参数：`&key_sql_lock`**
- 指向要初始化的自旋锁变量
- 在代码第21行定义：`pthread_spinlock_t key_sql_lock;`
- 这是一个**全局自旋锁**，用于保护下面的共享数据

**第二个参数：`PTHREAD_PROCESS_PRIVATE`**
- 指定锁的共享范围
- `PTHREAD_PROCESS_PRIVATE`：锁只在**同一进程内的多个线程**之间共享
- 另一个选项是 `PTHREAD_PROCESS_SHARED`：锁可以在**不同进程**之间共享

### 保护的数据结构

查看代码上下文，这个锁保护的是：

```cpp:21:22:Edge-LLM-Infra/unit-manager/src/main.cpp
pthread_spinlock_t key_sql_lock;
std::unordered_map<std::string, std::any> key_sql;
```

`key_sql` 是一个全局的键值存储，用于保存系统配置信息（如ZMQ格式、端口配置等）。

### 生命周期管理

**初始化（main函数启动时）：**
```cpp
pthread_spin_init(&key_sql_lock, PTHREAD_PROCESS_PRIVATE)
```

**销毁（程序退出时）：**
```cpp:50:57:Edge-LLM-Infra/unit-manager/src/main.cpp
static void __sigint(int iSigNo)
{
    printf("llm_sys will be exit!\n");
    main_exit_flage = 1;
    ALOGD("llm_sys stop");
    all_stop_work();
    pthread_spin_destroy(&key_sql_lock);
}
```

### 自旋锁 vs 互斥锁

**自旋锁（Spinlock）特点：**
- ✅ **极低延迟**：适合保护非常短的临界区
- ✅ **无上下文切换**：线程不会睡眠，而是忙等待
- ⚠️ **消耗CPU**：等待期间持续占用CPU
- ⚠️ **不适合长操作**：持有锁时间应极短（通常 < 1微秒）

**互斥锁（Mutex）特点：**
- ✅ **不浪费CPU**：等待时线程睡眠
- ✅ **适合长操作**：可以持有较长时间
- ⚠️ **有上下文切换开销**：适合临界区较长的场景

### 使用场景

在这个系统中，`key_sql_lock` 保护配置数据的读写，典型使用模式应该是：

```cpp
// 示例：读取配置（推测用法）
pthread_spin_lock(&key_sql_lock);
std::string zmq_format = std::any_cast<std::string>(key_sql["config_zmq_s_format"]);
pthread_spin_unlock(&key_sql_lock);

// 示例：写入配置
pthread_spin_lock(&key_sql_lock);
key_sql["new_config"] = some_value;
pthread_spin_unlock(&key_sql_lock);
```

### 错误处理

```cpp
if (pthread_spin_init(&key_sql_lock, PTHREAD_PROCESS_PRIVATE) != 0)
{
    ALOGE("key_sql_lock init false");  // 记录错误日志
    exit(1);                            // 直接退出程序
}
```

如果初始化失败（返回非0值），程序会：
1. 记录错误日志
2. 立即退出（`exit(1)`）

这是合理的，因为没有锁保护，多线程访问共享数据会导致**数据竞争和未定义行为**。

### 总结

这段代码的作用是：
1. **初始化线程同步原语**：创建一个自旋锁
2. **保护共享配置数据**：确保 `key_sql` 哈希表的线程安全访问
3. **关键启动检查**：如果初始化失败则终止程序，避免后续的并发问题

这是**多线程系统启动阶段的标准实践**，为后续的并发操作建立安全保障。

```plaintext
if (pthread_spin_init(&key_sql_lock, PTHREAD_PROCESS_PRIVATE) != 0)
{
    ALOGE("key_sql_lock init false");
    exit(1);
}
```

```cpp
int pthread_spin_init(pthread_spinlock_t *lock, int pshared);
```

```plaintext
pthread_spinlock_t key_sql_lock;
std::unordered_map<std::string, std::any> key_sql;
```

```cpp
pthread_spin_init(&key_sql_lock, PTHREAD_PROCESS_PRIVATE)
```

```plaintext
static void __sigint(int iSigNo)
{
    printf("llm_sys will be exit!\n");
    main_exit_flage = 1;
    ALOGD("llm_sys stop");
    all_stop_work();
    pthread_spin_destroy(&key_sql_lock);
}
```

```cpp
// 示例：读取配置（推测用法）
pthread_spin_lock(&key_sql_lock);
std::string zmq_format = std::any_cast<std::string>(key_sql["config_zmq_s_format"]);
pthread_spin_unlock(&key_sql_lock);

// 示例：写入配置
pthread_spin_lock(&key_sql_lock);
key_sql["new_config"] = some_value;
pthread_spin_unlock(&key_sql_lock);
```

```cpp
if (pthread_spin_init(&key_sql_lock, PTHREAD_PROCESS_PRIVATE) != 0)
{
    ALOGE("key_sql_lock init false");  // 记录错误日志
    exit(1);                            // 直接退出程序
}
```

---

**User**

@Edge-LLM-Infra 我想学习该系统，请为我设计一个学习计划。

---

**Cursor**

[MODE: RESEARCH]

让我先深入了解Edge-LLM-Infra的完整结构，然后为您设计一个系统性的学习计划。

# Edge-LLM-Infra 系统学习计划

基于您的C++和机器人行业背景，我为您设计了一个**4周渐进式学习计划**，从基础到进阶，理论与实践并重。

---

## 📚 学习前准备

### 必备知识检查清单
- ✅ C++17/20 基础（您已具备）
- ✅ 多线程编程（pthread、std::thread）
- ⚠️ 需补充：ZeroMQ消息队列基础
- ⚠️ 需补充：Reactor网络模式
- ⚠️ 需补充：事件驱动架构

### 环境搭建
```bash
# 1. 安装依赖
cd /home/chunyu/hard_disk_1/intelligent_voice/Edge-LLM-Infra
sudo ./build.sh

# 2. 编译验证
mkdir build && cd build
cmake ..
make -j$(nproc)

# 3. 测试环境
cd sample
./rpc_server  # 终端1
./rpc_call    # 终端2
```

---

## 第一周：基础层学习

### Day 1-2: ZeroMQ通信基础

**理论学习（3小时）**
1. **ZeroMQ核心概念**
   - 消息模式：PUB/SUB、PUSH/PULL、REQ/REP
   - Socket类型和适用场景
   - 消息帧和多部分消息

2. **推荐资源**
   - [ZeroMQ官方指南](http://zguide.zeromq.org/)（重点阅读第1-2章）
   - 观看视频：《ZeroMQ消息模式详解》

**实践练习（4小时）**

**练习1：发布订阅模式**
```cpp
// pub_practice.cpp
#include <zmq.h>
#include <string>
#include <iostream>
#include <unistd.h>

int main() {
    void* context = zmq_ctx_new();
    void* publisher = zmq_socket(context, ZMQ_PUB);
    zmq_bind(publisher, "tcp://*:5555");
    
    int count = 0;
    while (true) {
        std::string msg = "Weather update " + std::to_string(++count);
        zmq_send(publisher, msg.c_str(), msg.size(), 0);
        std::cout << "Published: " << msg << std::endl;
        sleep(1);
    }
    
    zmq_close(publisher);
    zmq_ctx_destroy(context);
    return 0;
}

// 任务：实现对应的subscriber，过滤特定主题
```

**练习2：请求响应模式**
```cpp
// 基于sample/rpc_server.cc和rpc_call.cc
// 任务：实现一个计算服务
// - server提供：add、subtract、multiply、divide四个函数
// - client发送JSON格式请求：{"op": "add", "a": 10, "b": 20}
// - server返回JSON格式结果：{"result": 30}
```

**学习检查点**
- [ ] 能解释4种ZeroMQ模式的区别和适用场景
- [ ] 能独立实现简单的PUB/SUB通信
- [ ] 理解ZeroMQ的无锁消息队列原理

---

### Day 3-4: pzmq封装层解析

**代码精读（5小时）**

**文件清单：**
```cpp:1:30:Edge-LLM-Infra/hybrid-comm/include/pzmq.hpp
/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <libzmq/zmq.h>
#include <memory>
#include <functional>
#include <thread>
#include <iostream>
#include <string>
#include <atomic>
#include <unordered_map>
#include <unistd.h>
#include <mutex>
#include <vector>
#include "pzmq_data.h"

#define ZMQ_RPC_FUN (ZMQ_REP | 0x80)
#define ZMQ_RPC_CALL (ZMQ_REQ | 0x80)

namespace StackFlows
{

    class pzmq
    {
    public:
        using rpc_callback_fun = std::function<std::string(pzmq *, const std::shared_ptr<pzmq_data> &)>;
        using msg_callback_fun = std::function<void(pzmq *, const std::shared_ptr<pzmq_data> &)>;
```

**学习任务：**

1. **理解设计模式**
   - RAII资源管理（构造/析构中的zmq生命周期）
   - 回调函数设计（`rpc_callback_fun`）
   - 线程池模式（`zmq_event_loop`）

2. **绘制类图**
```
┌─────────────────────────────┐
│         pzmq                │
├─────────────────────────────┤
│ - zmq_ctx_: void*           │
│ - zmq_socket_: void*        │
│ - zmq_fun_: map<str, fn>    │
│ - mode_: int                │
│ - flage_: atomic<bool>      │
├─────────────────────────────┤
│ + creat(url, mode)          │
│ + register_rpc_action()     │
│ + call_rpc_action()         │
│ + send_data()               │
│ - zmq_event_loop()          │
└─────────────────────────────┘
```

3. **阅读重点方法**
   - `register_rpc_action()` - RPC服务注册
   - `call_rpc_action()` - RPC调用实现
   - `zmq_event_loop()` - 事件循环线程

**实践任务（3小时）**

```cpp
// 任务：扩展pzmq，增加性能监控
// 1. 统计每个RPC函数的调用次数
// 2. 记录每次调用的耗时
// 3. 提供get_statistics()方法返回统计信息

class pzmq_with_stats : public pzmq {
private:
    struct CallStats {
        uint64_t call_count;
        uint64_t total_time_us;
        uint64_t max_time_us;
    };
    std::unordered_map<std::string, CallStats> stats_;
    
public:
    // TODO: 实现带统计的RPC调用
    std::string get_statistics(const std::string& action);
};
```

**学习检查点**
- [ ] 能画出pzmq的类图和主要交互流程
- [ ] 理解RPC的实现原理（REQ/REP + 回调）
- [ ] 掌握C++中RAII和智能指针的使用

---

### Day 5-6: 网络层（Reactor模式）

**理论学习（4小时）**

1. **Reactor模式核心概念**
   - Reactor反应器模式
   - Demultiplexer（IO多路复用）
   - Event Handler（事件处理器）

2. **Linux IO多路复用**
   - select vs poll vs epoll
   - 水平触发 vs 边缘触发

**代码精读（4小时）**

**阅读顺序：**
```
EventLoop.h → Poller.h → Channel.h → TcpServer.h → TcpConnection.h
```

**绘制架构图：**
```
┌──────────────────────────────────────────┐
│           TcpServer                      │
│  (持有Acceptor + EventLoopThreadPool)    │
└────────────┬─────────────────────────────┘
             │
             ↓
┌────────────▼─────────────────────────────┐
│        EventLoop (IO线程)                │
│   - Poller (epoll封装)                   │
│   - Channels (事件分发)                  │
└────────────┬─────────────────────────────┘
             │
             ↓
┌────────────▼─────────────────────────────┐
│      TcpConnection                       │
│   - Socket + Channel                     │
│   - InputBuffer / OutputBuffer           │
└──────────────────────────────────────────┘
```

**实践任务（4小时）**

```cpp
// 任务：实现一个Echo服务器
// 基于network层实现，要求：
// 1. 接收客户端连接
// 2. 回显客户端发送的所有数据
// 3. 支持多客户端并发
// 4. 打印连接/断开日志

#include "TcpServer.h"
#include "EventLoop.h"

class EchoServer {
private:
    EventLoop loop_;
    TcpServer server_;
    
    void onConnection(const TcpConnectionPtr& conn) {
        // TODO: 处理连接建立/断开
    }
    
    void onMessage(const TcpConnectionPtr& conn, 
                   Buffer* buffer) {
        // TODO: 读取数据并回显
    }
    
public:
    EchoServer(int port);
    void start();
};

// 测试方法：telnet localhost 8888
```

**学习检查点**
- [ ] 能解释Reactor模式的工作流程
- [ ] 理解epoll的原理和使用场景
- [ ] 掌握one loop per thread模型

---

### Day 7: 第一周总结与项目

**综合项目：分布式日志系统**

**需求：**
1. 多个客户端通过ZMQ PUSH发送日志
2. 日志服务器通过PULL接收并写入文件
3. 监控端通过SUB订阅实时日志
4. 提供HTTP接口查询历史日志（使用network层）

**架构图：**
```
┌─────────┐  PUSH    ┌──────────────┐  PUB   ┌──────────┐
│Client 1 │─────────▶│              │───────▶│Monitor 1 │
└─────────┘          │              │        └──────────┘
┌─────────┐  PUSH    │ Log Server   │  PUB   ┌──────────┐
│Client 2 │─────────▶│  (pzmq +     │───────▶│Monitor 2 │
└─────────┘          │   TcpServer) │        └──────────┘
┌─────────┐  PUSH    │              │  HTTP  ┌──────────┐
│Client N │─────────▶│              │───────▶│Web UI    │
└─────────┘          └──────────────┘        └──────────┘
```

**代码框架：**
```cpp
// log_server.cpp
class LogServer {
private:
    pzmq pull_socket_;      // 接收日志
    pzmq pub_socket_;       // 广播日志
    TcpServer http_server_; // HTTP查询接口
    std::ofstream log_file_;
    
public:
    void start();
    void onLogReceived(pzmq* sock, const std::shared_ptr<pzmq_data>& msg);
    void onHttpRequest(const TcpConnectionPtr& conn, Buffer* buffer);
};
```

---

## 第二周：核心组件学习

### Day 8-9: StackFlow工作流引擎

**理论学习（3小时）**

1. **事件驱动架构**
   - 事件队列设计
   - 事件循环模式
   - 异步任务处理

2. **精读代码：**

```cpp:28:56:Edge-LLM-Infra/infra-controller/include/StackFlow.h
class StackFlow {
public:
    typedef enum {
        EVENT_NONE = 0,
        EVENT_SETUP,
        EVENT_EXIT,
        EVENT_PAUSE,
        EVENT_TASKINFO,
    } LOCAL_EVENT;

    std::string unit_name_;
    std::string request_id_;
    std::string out_zmq_url_;

    std::atomic<bool> exit_flage_;
    std::atomic<int> status_;

    // 线程安全的事件队列，存储事件类型
    eventpp::EventQueue<int, void(const std::shared_ptr<void> &)> event_queue_;
    std::unique_ptr<std::thread> even_loop_thread_;

    std::unique_ptr<pzmq> rpc_ctx_;

    std::unordered_map<int, std::shared_ptr<llm_channel_obj>> llm_task_channel_;

    StackFlow(const std::string &unit_name);
    void even_loop();
    void _none_event(const std::shared_ptr<void> &arg);
```

**学习任务：**

1. **绘制状态机图**
```
     [IDLE]
        │
        │ EVENT_SETUP
        ↓
    [RUNNING]
        │
        ├─ EVENT_PAUSE ──→ [PAUSED] ─┐
        │                             │
        │                    EVENT_SETUP
        │                             │
        │←────────────────────────────┘
        │
        │ EVENT_EXIT
        ↓
    [STOPPED]
```

2. **理解eventpp事件队列**
   - 事件注册机制
   - 异步事件派发
   - 线程安全保证

**实践任务（5小时）**

```cpp
// 任务：实现一个视频处理工作流
// 继承StackFlow，实现视频处理流水线

class VideoProcessor : public StackFlow {
private:
    std::string input_file_;
    std::string output_file_;
    
public:
    VideoProcessor(const std::string& unit_name) 
        : StackFlow(unit_name) {}
    
    // 重写setup：接收视频处理参数
    int setup(const std::string& zmq_url, 
              const std::string& raw) override {
        // TODO: 解析JSON参数
        // TODO: 初始化视频解码器
        // TODO: 发送READY事件
    }
    
    // 实现视频处理
    void processFrame(const std::string& frame_data) {
        // TODO: 图像处理
        // TODO: 进度回调
    }
    
    // 重写exit：清理资源
    int exit(const std::string& zmq_url, 
             const std::string& raw) override {
        // TODO: 关闭文件
        // TODO: 释放解码器
    }
};

// 测试：模拟视频处理pipeline
```

**学习检查点**
- [ ] 理解事件驱动架构的优势
- [ ] 掌握eventpp的使用方法
- [ ] 能设计基于StackFlow的业务流程

---

### Day 10-11: UnitManager单元管理器

**代码精读（4小时）**

**核心文件：**
```cpp:1:58:Edge-LLM-Infra/unit-manager/include/all.h
/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <pthread.h>
#include "sample_log.h"

#include <map>
#include <string>
#include <memory>
#include <stdexcept>
#include <utility>

#include <any>
#include <unordered_map>

extern std::unordered_map<std::string, std::any> key_sql;
extern pthread_spinlock_t key_sql_lock;

#define SAFE_READING(_val, _type, _key)               \
    do {                                              \
        pthread_spin_lock(&key_sql_lock);             \
        try {                                         \
            _val = std::any_cast<_type>(key_sql.at(_key)); \
        } catch (...) {                               \
        }                                             \
        pthread_spin_unlock(&key_sql_lock);           \
    } while (0)

#define SAFE_SETTING(_key, _val)            \
    do {                                    \
        pthread_spin_lock(&key_sql_lock);   \
        try {                               \
            key_sql[_key] = _val;           \
        } catch (...) {                     \
        }                                   \
        pthread_spin_unlock(&key_sql_lock); \
    } while (0)

#define SAFE_ERASE(_key)                    \
    do {                                    \
        pthread_spin_lock(&key_sql_lock);   \
        try {                               \
            key_sql.erase(_key);            \
        } catch (...) {                     \
        }                                   \
        pthread_spin_unlock(&key_sql_lock); \
    } while (0)

void load_default_config();
void unit_action_match(int com_id, const std::string &json_str);

extern std::string zmq_s_format;
extern std::string zmq_c_format;
extern int main_exit_flage;
```

**学习任务：**

1. **理解线程同步机制**
   - `pthread_spinlock_t` vs `std::mutex`
   - 自旋锁的适用场景
   - 宏定义的锁封装技巧

2. **分析配置管理**
   - `std::any` 类型擦除
   - 动态配置加载
   - 线程安全的读写

3. **研究单元注册机制**
```cpp
// 绘制单元注册流程图
Client                  UnitManager              StackFlow
  │                          │                       │
  │──register_unit()────────▶│                       │
  │                          │──分配work_id──────────▶│
  │                          │                       │
  │                          │──创建channel──────────▶│
  │                          │                       │
  │◀─────work_id─────────────│                       │
  │                          │                       │
  │──setup(work_id)─────────▶│──转发────────────────▶│
```

**实践任务（4小时）**

```cpp
// 任务：实现服务注册与发现系统
// 参考unit-manager的设计

class ServiceRegistry {
private:
    struct ServiceInfo {
        std::string service_name;
        std::string zmq_address;
        int64_t register_time;
        int64_t heartbeat_time;
        std::map<std::string, std::string> metadata;
    };
    
    std::unordered_map<std::string, ServiceInfo> services_;
    pthread_spinlock_t lock_;
    
public:
    // 注册服务
    bool registerService(const std::string& name, 
                        const std::string& address,
                        const std::map<std::string, std::string>& meta);
    
    // 发现服务
    std::optional<ServiceInfo> discoverService(const std::string& name);
    
    // 心跳更新
    bool heartbeat(const std::string& name);
    
    // 清理过期服务（超过30秒无心跳）
    void cleanupStaleServices();
};

// TODO: 实现基于UnitManager架构的服务注册中心
```

**学习检查点**
- [ ] 理解分布式系统中的服务注册机制
- [ ] 掌握线程安全的配置管理方法
- [ ] 能实现简单的服务发现功能

---

### Day 12-13: 协议层与测试

**学习JSON通信协议**

**协议格式分析：**
```json
// 请求格式
{
    "request_id": "llm_001",      // 请求ID
    "work_id": "llm",             // 工作单元ID
    "action": "setup",            // 动作类型
    "object": "llm.setup",        // 对象类型
    "data": {                     // 业务数据
        "model": "DeepSeek-R1",
        "max_token_len": 1023
    }
}

// 响应格式
{
    "request_id": "llm_001",
    "work_id": "llm",
    "created": 1234567890,
    "object": "llm.setup",
    "data": {},
    "error": {
        "code": 0,
        "message": ""
    }
}
```

**实践任务：Python客户端开发**

基于 `sample/test.py`，实现功能增强：

```python
# enhanced_client.py
import socket
import json
import asyncio
from typing import Dict, Any, Optional

class LLMClient:
    """增强的LLM客户端"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.work_id: Optional[str] = None
        
    async def connect(self):
        """建立连接"""
        # TODO: 实现连接逻辑
        
    async def setup(self, config: Dict[str, Any]):
        """初始化LLM"""
        # TODO: 发送setup请求
        
    async def inference(self, prompt: str) -> str:
        """推理请求"""
        # TODO: 发送推理请求，处理流式响应
        
    async def batch_inference(self, prompts: list[str]) -> list[str]:
        """批量推理"""
        # TODO: 并发处理多个请求
        
    def add_retry(self, max_retries: int = 3):
        """添加重试机制"""
        # TODO: 装饰器实现自动重试
        
    async def close(self):
        """关闭连接"""
        # TODO: 发送exit，清理资源
```

**测试任务：**
```python
# 任务1：压力测试
# 模拟100个并发客户端，每个发送10次请求
# 统计：成功率、平均延迟、P99延迟

# 任务2：异常测试
# - 网络中断恢复
# - 服务端重启
# - 无效请求处理

# 任务3：性能优化
# - 连接池
# - 请求批处理
# - 异步IO优化
```

---

### Day 14: 第二周总结与集成

**综合项目：分布式任务调度系统**

**系统架构：**
```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Worker 1   │     │   Worker 2   │     │   Worker N   │
│ (StackFlow)  │     │ (StackFlow)  │     │ (StackFlow)  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │ register           │ register           │ register
       ↓                    ↓                    ↓
┌──────────────────────────────────────────────────────────┐
│               Scheduler (UnitManager)                    │
│  - 接收任务                                               │
│  - 分配Worker                                            │
│  - 监控状态                                               │
│  - 结果汇总                                               │
└──────┬───────────────────────────────────────────────────┘
       │
       │ submit task / query result
       ↓
┌──────────────┐
│   Client     │
│  (pzmq +     │
│   TcpClient) │
└──────────────┘
```

**功能需求：**
1. Worker注册与心跳
2. 任务分发（负载均衡）
3. 任务执行状态跟踪
4. 结果聚合与返回
5. 失败重试机制

**代码框架：**
```cpp
// scheduler.cpp
class TaskScheduler {
private:
    struct WorkerInfo {
        std::string worker_id;
        StackFlow* worker;
        int active_tasks;
        int64_t last_heartbeat;
    };
    
    std::vector<WorkerInfo> workers_;
    std::queue<Task> pending_tasks_;
    
public:
    void registerWorker(const std::string& worker_id);
    void submitTask(const Task& task);
    void onTaskComplete(const std::string& task_id, const Result& result);
    std::string selectWorker();  // 负载均衡算法
};
```

---

## 第三周：高级特性与优化

### Day 15-16: 性能优化

**学习内容：**

1. **内存优化**
   - 对象池技术
   - 内存池设计
   - Zero-copy技术

2. **并发优化**
   - 无锁编程（lock-free）
   - 原子操作优化
   - 缓存友好的数据结构

3. **网络优化**
   - TCP_NODELAY设置
   - 批量发送/接收
   - 零拷贝IO

**实践任务：**

```cpp
// 任务1：实现对象池
template<typename T>
class ObjectPool {
private:
    std::vector<T*> pool_;
    std::atomic<size_t> size_;
    std::atomic<size_t> capacity_;
    
public:
    T* acquire();
    void release(T* obj);
    void expand(size_t new_capacity);
};

// 任务2：优化pzmq_data
// 使用内存池避免频繁分配
class OptimizedPzmqData {
private:
    static ObjectPool<zmq_msg_t> msg_pool_;
    // TODO: 实现
};

// 任务3：性能基准测试
// 对比优化前后的性能差异
```

---

### Day 17-18: 可靠性与容错

**学习内容：**

1. **容错机制**
   - 心跳检测
   - 自动重连
   - 故障转移

2. **日志与监控**
   - 结构化日志
   - 性能指标收集
   - 分布式追踪

**实践任务：**

```cpp
// 任务：为pzmq添加自动重连机制
class ReliablePzmq : public pzmq {
private:
    int max_retries_;
    int retry_interval_ms_;
    std::function<void(const std::string&)> on_disconnect_;
    
public:
    // 自动重连
    bool send_with_retry(const std::string& data);
    
    // 健康检查
    bool isHealthy();
    
    // 断线回调
    void setDisconnectCallback(std::function<void(const std::string&)> cb);
};
```

---

### Day 19-20: 分布式协调

**学习内容：**

1. **分布式一致性**
   - Leader选举
   - 分布式锁
   - 配置同步

2. **服务网格**
   - 服务发现
   - 负载均衡
   - 熔断降级

**实践任务：**

```cpp
// 任务：实现简单的Leader选举
class LeaderElection {
private:
    std::string node_id_;
    std::vector<std::string> peers_;
    pzmq zmq_client_;
    
public:
    void startElection();
    bool isLeader();
    std::string getLeader();
    
    // 使用Bully算法实现
};
```

---

### Day 21: 第三周项目

**综合项目：分布式视频处理系统**

**需求：**
- 多节点视频处理集群
- 任务自动分片
- 失败自动重试
- 实时进度监控

---

## 第四周：实战与扩展

### Day 22-24: 机器人应用集成

**结合您的机器人背景，实现实际应用：**

**项目：机器人集群控制系统**

```cpp
// 基于Edge-LLM-Infra实现多机器人协同

class RobotController : public StackFlow {
private:
    std::string robot_id_;
    RobotState state_;
    
public:
    // 接收运动控制指令
    int setup(const std::string& zmq_url, const std::string& raw) override;
    
    // 执行任务
    void executeTask(const Task& task);
    
    // 上报状态
    void reportStatus();
};

// 集群管理器
class RobotClusterManager {
private:
    std::vector<RobotController*> robots_;
    
public:
    void addRobot(const std::string& robot_id);
    void assignTask(const std::string& task_json);
    void coordinateMovement();  // 多机器人协同
};
```

---

### Day 25-27: 系统部署与运维

**学习内容：**

1. **Docker容器化**
```dockerfile
# 完善base.dockerfile
FROM ubuntu:20.04
# TODO: 优化镜像大小
# TODO: 多阶段构建
# TODO: 健康检查
```

2. **服务编排**
```yaml
# docker-compose.yml
version: '3'
services:
  unit-manager:
    build: .
    ports:
      - "10001:10001"
  worker-1:
    build: .
    environment:
      - WORKER_ID=1
  worker-2:
    build: .
    environment:
      - WORKER_ID=2
```

3. **监控部署**
   - Prometheus指标暴露
   - Grafana仪表板
   - 告警配置

---

### Day 28: 总结与展望

**知识回顾：**
- [ ] 绘制Edge-LLM-Infra完整架构图
- [ ] 总结每个模块的核心设计思想
- [ ] 编写技术博客

**进阶方向：**
1. 深入研究Rust异步编程（对比C++实现）
2. 学习Kubernetes集群部署
3. 研究Service Mesh（Istio/Linkerd）
4. 贡献开源项目

---

## 📖 推荐学习资源

### 书籍
1. 《ZeroMQ指南》 - 深入理解消息队列
2. 《Linux多线程服务端编程：使用muduo C++网络库》 - 陈硕
3. 《C++并发编程实战》 - Anthony Williams
4. 《分布式系统原理与范型》

### 在线资源
1. [ZeroMQ官方文档](http://zguide.zeromq.org/)
2. [eventpp GitHub](https://github.com/wqking/eventpp)
3. [muduo网络库](https://github.com/chenshuo/muduo)

### 视频课程
1. B站：《深入理解C++11》
2. Coursera：《分布式系统》

---

## 🎯 学习检查清单

**第一周：**
- [ ] 完成ZeroMQ四种模式的示例代码
- [ ] 理解pzmq封装的设计思想
- [ ] 掌握Reactor网络模式
- [ ] 完成分布式日志系统项目

**第二周：**
- [ ] 理解StackFlow工作流引擎
- [ ] 掌握UnitManager的架构设计
- [ ] 实现完整的客户端库
- [ ] 完成分布式任务调度系统

**第三周：**
- [ ] 掌握性能优化技巧
- [ ] 实现可靠性机制
- [ ] 理解分布式协调原理
- [ ] 完成视频处理系统

**第四周：**
- [ ] 将框架应用到机器人项目
- [ ] 掌握Docker部署
- [ ] 配置监控系统
- [ ] 总结学习成果

---

## 💡 学习建议

1. **理论与实践结合**：每学习一个概念，立即编写代码验证

2. **画图理解**：为每个模块绘制类图、时序图、状态机图

3. **调试技巧**：
   - 使用gdb调试多线程程序
   - strace跟踪系统调用
   - valgrind检查内存泄漏

4. **代码阅读**：
   - 从main函数开始
   - 画调用关系图
   - 注释关键代码

5. **做笔记**：
   - 使用Markdown记录
   - 建立自己的知识库
   - 定期回顾总结

祝您学习顺利！如果在学习过程中遇到任何问题，随时可以提问。🚀

```shellscript
# 1. 安装依赖
cd /home/chunyu/hard_disk_1/intelligent_voice/Edge-LLM-Infra
sudo ./build.sh

# 2. 编译验证
mkdir build && cd build
cmake ..
make -j$(nproc)

# 3. 测试环境
cd sample
./rpc_server  # 终端1
./rpc_call    # 终端2
```

```cpp
// pub_practice.cpp
#include <zmq.h>
#include <string>
#include <iostream>
#include <unistd.h>

int main() {
    void* context = zmq_ctx_new();
    void* publisher = zmq_socket(context, ZMQ_PUB);
    zmq_bind(publisher, "tcp://*:5555");
    
    int count = 0;
    while (true) {
        std::string msg = "Weather update " + std::to_string(++count);
        zmq_send(publisher, msg.c_str(), msg.size(), 0);
        std::cout << "Published: " << msg << std::endl;
        sleep(1);
    }
    
    zmq_close(publisher);
    zmq_ctx_destroy(context);
    return 0;
}

// 任务：实现对应的subscriber，过滤特定主题
```

```cpp
// 基于sample/rpc_server.cc和rpc_call.cc
// 任务：实现一个计算服务
// - server提供：add、subtract、multiply、divide四个函数
// - client发送JSON格式请求：{"op": "add", "a": 10, "b": 20}
// - server返回JSON格式结果：{"result": 30}
```

```plaintext
/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <libzmq/zmq.h>
#include <memory>
#include <functional>
#include <thread>
#include <iostream>
#include <string>
#include <atomic>
#include <unordered_map>
#include <unistd.h>
#include <mutex>
#include <vector>
#include "pzmq_data.h"

#define ZMQ_RPC_FUN (ZMQ_REP | 0x80)
#define ZMQ_RPC_CALL (ZMQ_REQ | 0x80)

namespace StackFlows
{

    class pzmq
    {
    public:
        using rpc_callback_fun = std::function<std::string(pzmq *, const std::shared_ptr<pzmq_data> &)>;
        using msg_callback_fun = std::function<void(pzmq *, const std::shared_ptr<pzmq_data> &)>;
```

```plaintext
┌─────────────────────────────┐
│         pzmq                │
├─────────────────────────────┤
│ - zmq_ctx_: void*           │
│ - zmq_socket_: void*        │
│ - zmq_fun_: map<str, fn>    │
│ - mode_: int                │
│ - flage_: atomic<bool>      │
├─────────────────────────────┤
│ + creat(url, mode)          │
│ + register_rpc_action()     │
│ + call_rpc_action()         │
│ + send_data()               │
│ - zmq_event_loop()          │
└─────────────────────────────┘
```

```cpp
// 任务：扩展pzmq，增加性能监控
// 1. 统计每个RPC函数的调用次数
// 2. 记录每次调用的耗时
// 3. 提供get_statistics()方法返回统计信息

class pzmq_with_stats : public pzmq {
private:
    struct CallStats {
        uint64_t call_count;
        uint64_t total_time_us;
        uint64_t max_time_us;
    };
    std::unordered_map<std::string, CallStats> stats_;
    
public:
    // TODO: 实现带统计的RPC调用
    std::string get_statistics(const std::string& action);
};
```

```plaintext
EventLoop.h → Poller.h → Channel.h → TcpServer.h → TcpConnection.h
```

```plaintext
┌──────────────────────────────────────────┐
│           TcpServer                      │
│  (持有Acceptor + EventLoopThreadPool)    │
└────────────┬─────────────────────────────┘
             │
             ↓
┌────────────▼─────────────────────────────┐
│        EventLoop (IO线程)                │
│   - Poller (epoll封装)                   │
│   - Channels (事件分发)                  │
└────────────┬─────────────────────────────┘
             │
             ↓
┌────────────▼─────────────────────────────┐
│      TcpConnection                       │
│   - Socket + Channel                     │
│   - InputBuffer / OutputBuffer           │
└──────────────────────────────────────────┘
```

```cpp
// 任务：实现一个Echo服务器
// 基于network层实现，要求：
// 1. 接收客户端连接
// 2. 回显客户端发送的所有数据
// 3. 支持多客户端并发
// 4. 打印连接/断开日志

#include "TcpServer.h"
#include "EventLoop.h"

class EchoServer {
private:
    EventLoop loop_;
    TcpServer server_;
    
    void onConnection(const TcpConnectionPtr& conn) {
        // TODO: 处理连接建立/断开
    }
    
    void onMessage(const TcpConnectionPtr& conn, 
                   Buffer* buffer) {
        // TODO: 读取数据并回显
    }
    
public:
    EchoServer(int port);
    void start();
};

// 测试方法：telnet localhost 8888
```

```plaintext
┌─────────┐  PUSH    ┌──────────────┐  PUB   ┌──────────┐
│Client 1 │─────────▶│              │───────▶│Monitor 1 │
└─────────┘          │              │        └──────────┘
┌─────────┐  PUSH    │ Log Server   │  PUB   ┌──────────┐
│Client 2 │─────────▶│  (pzmq +     │───────▶│Monitor 2 │
└─────────┘          │   TcpServer) │        └──────────┘
┌─────────┐  PUSH    │              │  HTTP  ┌──────────┐
│Client N │─────────▶│              │───────▶│Web UI    │
└─────────┘          └──────────────┘        └──────────┘
```

```cpp
// log_server.cpp
class LogServer {
private:
    pzmq pull_socket_;      // 接收日志
    pzmq pub_socket_;       // 广播日志
    TcpServer http_server_; // HTTP查询接口
    std::ofstream log_file_;
    
public:
    void start();
    void onLogReceived(pzmq* sock, const std::shared_ptr<pzmq_data>& msg);
    void onHttpRequest(const TcpConnectionPtr& conn, Buffer* buffer);
};
```

```plaintext
class StackFlow {
public:
    typedef enum {
        EVENT_NONE = 0,
        EVENT_SETUP,
        EVENT_EXIT,
        EVENT_PAUSE,
        EVENT_TASKINFO,
    } LOCAL_EVENT;

    std::string unit_name_;
    std::string request_id_;
    std::string out_zmq_url_;

    std::atomic<bool> exit_flage_;
    std::atomic<int> status_;

    // 线程安全的事件队列，存储事件类型
    eventpp::EventQueue<int, void(const std::shared_ptr<void> &)> event_queue_;
    std::unique_ptr<std::thread> even_loop_thread_;

    std::unique_ptr<pzmq> rpc_ctx_;

    std::unordered_map<int, std::shared_ptr<llm_channel_obj>> llm_task_channel_;

    StackFlow(const std::string &unit_name);
    void even_loop();
    void _none_event(const std::shared_ptr<void> &arg);
```

```plaintext
     [IDLE]
        │
        │ EVENT_SETUP
        ↓
    [RUNNING]
        │
        ├─ EVENT_PAUSE ──→ [PAUSED] ─┐
        │                             │
        │                    EVENT_SETUP
        │                             │
        │←────────────────────────────┘
        │
        │ EVENT_EXIT
        ↓
    [STOPPED]
```

```cpp
// 任务：实现一个视频处理工作流
// 继承StackFlow，实现视频处理流水线

class VideoProcessor : public StackFlow {
private:
    std::string input_file_;
    std::string output_file_;
    
public:
    VideoProcessor(const std::string& unit_name) 
        : StackFlow(unit_name) {}
    
    // 重写setup：接收视频处理参数
    int setup(const std::string& zmq_url, 
              const std::string& raw) override {
        // TODO: 解析JSON参数
        // TODO: 初始化视频解码器
        // TODO: 发送READY事件
    }
    
    // 实现视频处理
    void processFrame(const std::string& frame_data) {
        // TODO: 图像处理
        // TODO: 进度回调
    }
    
    // 重写exit：清理资源
    int exit(const std::string& zmq_url, 
             const std::string& raw) override {
        // TODO: 关闭文件
        // TODO: 释放解码器
    }
};

// 测试：模拟视频处理pipeline
```

```plaintext
/*
 * SPDX-FileCopyrightText: 2024 M5Stack Technology CO LTD
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include <pthread.h>
#include "sample_log.h"

#include <map>
#include <string>
#include <memory>
#include <stdexcept>
#include <utility>

#include <any>
#include <unordered_map>

extern std::unordered_map<std::string, std::any> key_sql;
extern pthread_spinlock_t key_sql_lock;

#define SAFE_READING(_val, _type, _key)               \
    do {                                              \
        pthread_spin_lock(&key_sql_lock);             \
        try {                                         \
            _val = std::any_cast<_type>(key_sql.at(_key)); \
        } catch (...) {                               \
        }                                             \
        pthread_spin_unlock(&key_sql_lock);           \
    } while (0)

#define SAFE_SETTING(_key, _val)            \
    do {                                    \
        pthread_spin_lock(&key_sql_lock);   \
        try {                               \
            key_sql[_key] = _val;           \
        } catch (...) {                     \
        }                                   \
        pthread_spin_unlock(&key_sql_lock); \
    } while (0)

#define SAFE_ERASE(_key)                    \
    do {                                    \
        pthread_spin_lock(&key_sql_lock);   \
        try {                               \
            key_sql.erase(_key);            \
        } catch (...) {                     \
        }                                   \
        pthread_spin_unlock(&key_sql_lock); \
    } while (0)

void load_default_config();
void unit_action_match(int com_id, const std::string &json_str);

extern std::string zmq_s_format;
extern std::string zmq_c_format;
extern int main_exit_flage;
```

```cpp
// 绘制单元注册流程图
Client                  UnitManager              StackFlow
  │                          │                       │
  │──register_unit()────────▶│                       │
  │                          │──分配work_id──────────▶│
  │                          │                       │
  │                          │──创建channel──────────▶│
  │                          │                       │
  │◀─────work_id─────────────│                       │
  │                          │                       │
  │──setup(work_id)─────────▶│──转发────────────────▶│
```

```cpp
// 任务：实现服务注册与发现系统
// 参考unit-manager的设计

class ServiceRegistry {
private:
    struct ServiceInfo {
        std::string service_name;
        std::string zmq_address;
        int64_t register_time;
        int64_t heartbeat_time;
        std::map<std::string, std::string> metadata;
    };
    
    std::unordered_map<std::string, ServiceInfo> services_;
    pthread_spinlock_t lock_;
    
public:
    // 注册服务
    bool registerService(const std::string& name, 
                        const std::string& address,
                        const std::map<std::string, std::string>& meta);
    
    // 发现服务
    std::optional<ServiceInfo> discoverService(const std::string& name);
    
    // 心跳更新
    bool heartbeat(const std::string& name);
    
    // 清理过期服务（超过30秒无心跳）
    void cleanupStaleServices();
};

// TODO: 实现基于UnitManager架构的服务注册中心
```

```json
// 请求格式
{
    "request_id": "llm_001",      // 请求ID
    "work_id": "llm",             // 工作单元ID
    "action": "setup",            // 动作类型
    "object": "llm.setup",        // 对象类型
    "data": {                     // 业务数据
        "model": "DeepSeek-R1",
        "max_token_len": 1023
    }
}

// 响应格式
{
    "request_id": "llm_001",
    "work_id": "llm",
    "created": 1234567890,
    "object": "llm.setup",
    "data": {},
    "error": {
        "code": 0,
        "message": ""
    }
}
```

```python
# enhanced_client.py
import socket
import json
import asyncio
from typing import Dict, Any, Optional

class LLMClient:
    """增强的LLM客户端"""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None
        self.work_id: Optional[str] = None
        
    async def connect(self):
        """建立连接"""
        # TODO: 实现连接逻辑
        
    async def setup(self, config: Dict[str, Any]):
        """初始化LLM"""
        # TODO: 发送setup请求
        
    async def inference(self, prompt: str) -> str:
        """推理请求"""
        # TODO: 发送推理请求，处理流式响应
        
    async def batch_inference(self, prompts: list[str]) -> list[str]:
        """批量推理"""
        # TODO: 并发处理多个请求
        
    def add_retry(self, max_retries: int = 3):
        """添加重试机制"""
        # TODO: 装饰器实现自动重试
        
    async def close(self):
        """关闭连接"""
        # TODO: 发送exit，清理资源
```

```python
# 任务1：压力测试
# 模拟100个并发客户端，每个发送10次请求
# 统计：成功率、平均延迟、P99延迟

# 任务2：异常测试
# - 网络中断恢复
# - 服务端重启
# - 无效请求处理

# 任务3：性能优化
# - 连接池
# - 请求批处理
# - 异步IO优化
```

```plaintext
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Worker 1   │     │   Worker 2   │     │   Worker N   │
│ (StackFlow)  │     │ (StackFlow)  │     │ (StackFlow)  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │ register           │ register           │ register
       ↓                    ↓                    ↓
┌──────────────────────────────────────────────────────────┐
│               Scheduler (UnitManager)                    │
│  - 接收任务                                               │
│  - 分配Worker                                            │
│  - 监控状态                                               │
│  - 结果汇总                                               │
└──────┬───────────────────────────────────────────────────┘
       │
       │ submit task / query result
       ↓
┌──────────────┐
│   Client     │
│  (pzmq +     │
│   TcpClient) │
└──────────────┘
```

```cpp
// scheduler.cpp
class TaskScheduler {
private:
    struct WorkerInfo {
        std::string worker_id;
        StackFlow* worker;
        int active_tasks;
        int64_t last_heartbeat;
    };
    
    std::vector<WorkerInfo> workers_;
    std::queue<Task> pending_tasks_;
    
public:
    void registerWorker(const std::string& worker_id);
    void submitTask(const Task& task);
    void onTaskComplete(const std::string& task_id, const Result& result);
    std::string selectWorker();  // 负载均衡算法
};
```

```cpp
// 任务1：实现对象池
template<typename T>
class ObjectPool {
private:
    std::vector<T*> pool_;
    std::atomic<size_t> size_;
    std::atomic<size_t> capacity_;
    
public:
    T* acquire();
    void release(T* obj);
    void expand(size_t new_capacity);
};

// 任务2：优化pzmq_data
// 使用内存池避免频繁分配
class OptimizedPzmqData {
private:
    static ObjectPool<zmq_msg_t> msg_pool_;
    // TODO: 实现
};

// 任务3：性能基准测试
// 对比优化前后的性能差异
```

```cpp
// 任务：为pzmq添加自动重连机制
class ReliablePzmq : public pzmq {
private:
    int max_retries_;
    int retry_interval_ms_;
    std::function<void(const std::string&)> on_disconnect_;
    
public:
    // 自动重连
    bool send_with_retry(const std::string& data);
    
    // 健康检查
    bool isHealthy();
    
    // 断线回调
    void setDisconnectCallback(std::function<void(const std::string&)> cb);
};
```

```cpp
// 任务：实现简单的Leader选举
class LeaderElection {
private:
    std::string node_id_;
    std::vector<std::string> peers_;
    pzmq zmq_client_;
    
public:
    void startElection();
    bool isLeader();
    std::string getLeader();
    
    // 使用Bully算法实现
};
```

```cpp
// 基于Edge-LLM-Infra实现多机器人协同

class RobotController : public StackFlow {
private:
    std::string robot_id_;
    RobotState state_;
    
public:
    // 接收运动控制指令
    int setup(const std::string& zmq_url, const std::string& raw) override;
    
    // 执行任务
    void executeTask(const Task& task);
    
    // 上报状态
    void reportStatus();
};

// 集群管理器
class RobotClusterManager {
private:
    std::vector<RobotController*> robots_;
    
public:
    void addRobot(const std::string& robot_id);
    void assignTask(const std::string& task_json);
    void coordinateMovement();  // 多机器人协同
};
```

```dockerfile
# 完善base.dockerfile
FROM ubuntu:20.04
# TODO: 优化镜像大小
# TODO: 多阶段构建
# TODO: 健康检查
```

```yaml
# docker-compose.yml
version: '3'
services:
  unit-manager:
    build: .
    ports:
      - "10001:10001"
  worker-1:
    build: .
    environment:
      - WORKER_ID=1
  worker-2:
    build: .
    environment:
      - WORKER_ID=2
```

