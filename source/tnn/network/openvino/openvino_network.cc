// Copyright 2019 Tencent. All Rights Reserved

#include "openvino_network.h"
#include <string.h>

#include <ie_extension.h>

#include "tnn/core/blob_int8.h"
#include "tnn/core/profile.h"
#include "tnn/interpreter/default_model_interpreter.h"
#include "tnn/interpreter/layer_param.h"
#include "tnn/interpreter/layer_resource_generator.h"
#include "tnn/optimizer/net_optimizer_manager.h"
#include "tnn/utils/blob_dump_utils.h"
#include "tnn/utils/blob_transfer_utils.h"
#include "tnn/utils/dims_utils.h"
#include "tnn/utils/string_utils_inner.h"
#include "tnn/extern_wrapper/foreign_blob.h"
#include "tnn/extern_wrapper/foreign_tensor.h"
#include "tnn/network/openvino/layer_builder/openvino_layer_builder.h"
#include "tnn/network/openvino/openvino_types.h"

#include "tnn/network/openvino/custom_layer/custom_implementation.h"
#include "tnn/network/openvino/utils.h"

namespace TNN_NS {

NetworkImplFactoryRegister<NetworkImplFactory<OpenVINONetwork_>> g_network_impl_openvino_ngraph_factory_register(NETWORK_TYPE_OPENVINO);

OpenVINONetwork_::~OpenVINONetwork_() {
    DeInit();
}

Status OpenVINONetwork_::Init(NetworkConfig &net_config, ModelConfig &model_config,
                            AbstractModelInterpreter* interpreter,
                            InputShapesMap min_inputs_shape, InputShapesMap max_inputs_shape, bool enable_const_folder) {

    Status ret  = TNN_OK;

    // RETURN_ON_NEQ(DefaultNetwork::Init(net_config, model_config, interpreter, inputs_shape), TNN_OK);
    DefaultModelInterpreter *default_interpreter = dynamic_cast<DefaultModelInterpreter *>(interpreter);
    CHECK_PARAM_NULL(default_interpreter);

    NetStructure *net_structure = default_interpreter->GetNetStructure();
    NetResource *net_resource   = default_interpreter->GetNetResource();

    if (net_structure == NULL || net_resource == NULL) {
        LOGE("ERROR: network_ is nil, network_type may not support\n");
        return Status(TNNERR_NULL_PARAM, "network_ is nil, network_type may not support");
    }

    device_ = GetDevice(net_config.device_type);
    if (device_ == NULL) {
        return TNNERR_DEVICE_NOT_SUPPORT;
    }

    context_ = device_->CreateContext(net_config.device_id);
    if (context_ == NULL) {
        return TNNERR_DEVICE_CONTEXT_CREATE;
    }

    /*
     * The NetOptimizeManager holds a list of network optimization processes.
     * The optimization process may change the network structure accoundingly.
     * eg. fuse conv+bn, conv+relu.
     */
    {
        // use mutex to protect net_resource and net_structure in multi-thread
        std::unique_lock<std::mutex> lck(optimize_mtx_);
        ret = optimizer::NetOptimizerManager::Optimize(net_structure, net_resource, net_config);
        if (ret != TNN_OK) {
            return ret;
        }
    }

    blob_manager_ = new BlobManager(device_);
    ret = blob_manager_->Init(net_config, net_structure, max_inputs_shape, GetNetResourceDataType(net_resource));

    //set inputnode
    RETURN_ON_NEQ(SetNetInputNode(), TNN_OK);
    //init layers and nodes
    RETURN_ON_NEQ(InitLayers(net_structure, net_resource), TNN_OK);

    // build ngraph network
    RETURN_ON_NEQ(BuildNgraphNetwork(net_structure), TNN_OK);
    //////////////////////////////////////////////////////////////
    std::map<std::string, ov::Any> config = {
        {CONFIG_KEY(CPU_THREADS_NUM), "1"},
        {CONFIG_KEY(CPU_THROUGHPUT_STREAMS), "0"},
        {CONFIG_KEY(CPU_BIND_THREAD), "NO"},
    };

    ie_.set_property("CPU", config);

    InferenceEngine::IExtensionPtr extensionPtr;

    extensionPtr = std::make_shared<CustomOpenvinoLayerManager>();
    ie_.add_extension(extensionPtr);

    return Reshape(max_inputs_shape);
}

Status OpenVINONetwork_::SetNetInputNode() {
    BlobMap blob_map;
    blob_manager_->GetAllInputBlobs(blob_map);

    for(auto it : blob_map) {
        std::string input_name = it.first; 
        BlobDesc blob_desc = it.second->GetBlobDesc();

        ngraph::Shape  ngraph_input_shape;
        for(auto d : blob_desc.dims) {
            ngraph_input_shape.push_back(d);
        }

        std::shared_ptr<ngraph::op::Parameter> input_node = 
                std::make_shared<ngraph::op::Parameter>(ConvertToOVDataType(blob_desc.data_type), ngraph::Shape(ngraph_input_shape));
        input_node->set_friendly_name(input_name);

        auto foreign_blob = new ForeignBlob(it.second);
        foreign_blob->SetForeignTensor(std::make_shared<OpenvinoTensor>(input_node));

        blob_manager_->ReplaceBlob(input_name, foreign_blob);
    }

    return TNN_OK;
}

Status OpenVINONetwork_::BuildNgraphNetwork(NetStructure *net_structure) {
    ov::ParameterVector input_nodes;
    for(auto it : net_structure->inputs_shape_map) {
        auto name = it.first;
        auto input_tensor = dynamic_cast<ForeignBlob*>(blob_manager_->GetBlob(name))->GetForeignTensor();
        auto input_openvino_tensor = std::dynamic_pointer_cast<OpenvinoTensor>(input_tensor);
	input_openvino_tensor->GetNode()->set_friendly_name(name);
	input_nodes.push_back(std::dynamic_pointer_cast<ov::opset8::Parameter>(input_openvino_tensor->GetNode()));
    }

    ov::NodeVector output_nodes;
    for (auto name : net_structure->outputs) {
        auto output_tensor = dynamic_cast<ForeignBlob*>(blob_manager_->GetBlob(name))->GetForeignTensor();
        auto output_openvino_tensor = std::dynamic_pointer_cast<OpenvinoTensor>(output_tensor);
	auto result_node = std::make_shared<ov::opset8::Result>(output_openvino_tensor->GetNode());
	result_node->set_friendly_name(name);
        output_nodes.push_back(result_node);
    } 
    
    network_ = std::make_shared<ov::Model>(output_nodes, input_nodes, "net");

    return TNN_OK;
}

Status OpenVINONetwork_::GetForwardMemorySize(int &memory_size) {
    memory_size = 0;
    return TNN_OK;
}

Status OpenVINONetwork_::SetForwardMemory(void *memory) {   
    return TNN_OK;
}

Status OpenVINONetwork_::GetAllInputBlobs(BlobMap &blobs) {
    blobs = input_blob_map_;
    return TNN_OK;
}

Status OpenVINONetwork_::GetAllOutputBlobs(BlobMap &blobs) {
    blobs = output_blob_map_;
    // if tnn blob != openvino blob dims, use min(tnn_blob, openvino_blob)
    // eg. detection_output layer
    for (auto iter : blobs) {
        auto tnn_blob_dims = blob_manager_->GetBlob(iter.first)->GetBlobDesc().dims;
        auto openvino_blob_dims = iter.second->GetBlobDesc().dims;
        iter.second->GetBlobDesc().dims = DimsVectorUtils::Min(tnn_blob_dims, openvino_blob_dims);
    }
    return TNN_OK;
}

Status OpenVINONetwork_::Reshape(const InputShapesMap &inputs) {
    RETURN_ON_NEQ(DefaultNetwork::Reshape(inputs), TNN_OK);

    std::map<std::size_t, ov::PartialShape> network_shapes;
    for (const ov::Output<ov::Node>& input : network_->inputs()) {
        ov::PartialShape pshape = input.get_partial_shape();
	std::string input_name = input.get_node()->get_friendly_name();
	std::size_t input_idx = input.get_index();
	network_shapes.emplace(std::make_pair(input_idx, pshape));
    }

    std::size_t idx = 0;
    for(auto item : inputs) {
	if (network_shapes.find(idx) == network_shapes.end()) {
	    return TNNERR_PARAM_ERR;
	}
        if (item.second.size() != network_shapes.find(idx)->second.size()) {
            return TNNERR_PARAM_ERR;
        }
	
	std::vector<ov::Dimension> dimensions;
        for(int i=0;i<item.second.size();i++) {
	    ov::Dimension dimension = ov::Dimension(item.second[i]);
	    dimensions.push_back(dimension);
        }
	ov::PartialShape input_shape = ov::PartialShape(dimensions);
	network_shapes[idx] = input_shape;
	idx++;
    }

    network_->reshape(network_shapes);

    executable_network_ = ie_.compile_model(network_, "CPU");
    infer_request_ = executable_network_.create_infer_request();
    
    const std::vector<ov::Output<const ov::Node>>& inputs_info = executable_network_.inputs();
    for (int i = 0; i < inputs_info.size(); i++) {
	ov::Output<const ov::Node> item = inputs_info[i];

	std::string key = item.get_node()->get_friendly_name();
	std::size_t idx = item.get_index();

	ov::Tensor tensor = infer_request_.get_input_tensor(idx);
	const std::shared_ptr<ov::descriptor::Tensor> tensor_ptr = item.get_tensor_ptr();
	
        BlobDesc desc;
        desc.data_format = DATA_FORMAT_NCHW;
	desc.name = key;
        desc.device_type = DEVICE_X86;
	const ov::element::Type& element_type = tensor_ptr->get_element_type();
	desc.data_type = ConvertOVPrecisionToDataType(element_type);

	const ov::Shape dims = tensor_ptr->get_shape();
        for(int index = 0; index<dims.size(); index++) {
            desc.dims.push_back(dims[index]);
        }

        BlobHandle handle;
	handle.base = tensor.data<ov::element_type_traits<ov::element::Type_t::f32>::value_type>();

        if (input_blob_map_.find(key) != input_blob_map_.end())  {
            input_blob_map_[key]->SetBlobDesc(desc);
            input_blob_map_[key]->SetHandle(handle);
        } else {
            input_blob_map_[key] = new Blob(desc, handle);
        }
    }
    const std::vector<ov::Output<const ov::Node>>& outputs_info = executable_network_.outputs();
    for (int i = 0; i < outputs_info.size(); i++) {
        ov::Output<const ov::Node> item = outputs_info[i];
        //std::string key = item.first;
	std::string key = item.get_node()->get_friendly_name();
	std::size_t idx = item.get_index();

	ov::Tensor tensor = infer_request_.get_output_tensor(idx);
	const std::shared_ptr<ov::descriptor::Tensor> tensor_ptr = item.get_tensor_ptr();

        BlobDesc desc;
        desc.data_format = DATA_FORMAT_NCHW;
        desc.name = key;
        desc.device_type = DEVICE_X86;
        
	const ov::element::Type& element_type = tensor_ptr->get_element_type();
	desc.data_type = ConvertOVPrecisionToDataType(element_type);
	const ov::Shape dims = tensor_ptr->get_shape();
        for(int index = 0; index<dims.size(); index++) {
            desc.dims.push_back(dims[index]);
        }
        BlobHandle handle;
	handle.base = tensor.data<ov::element_type_traits<ov::element::Type_t::f32>::value_type>();
        if (output_blob_map_.find(key) != output_blob_map_.end())  {
            output_blob_map_[key]->SetBlobDesc(desc);
            output_blob_map_[key]->SetHandle(handle);
        } else {
            output_blob_map_[key] = new Blob(desc, handle);
        }
    }

    return TNN_OK;
}

/*
 * InitLayerBuilders funcion does the following things:
 *  1. Set Blob type accordingly.
 *  2. Set data_tyep accordingly.
 *  3. Infer the blob shapes.
 *  4. Check the weights required.
 *  5. Create Layer Builders.
 */
Status OpenVINONetwork_::InitLayers(NetStructure *net_structure, NetResource *net_resource) {
    Status ret = TNN_OK;

    auto const_blobs = net_resource->constant_map;
    for (auto layer_info : net_structure->layers) {
        std::vector<std::string> &input_names = layer_info->inputs;
        for (auto name : input_names) {
            auto blob = blob_manager_->GetBlob(name);
            if (const_blobs.find(name) != const_blobs.end()) {
                blob->GetBlobDesc().data_type = const_blobs[name]->GetDataType();
                if (runtime_model_ == RUNTIME_MODE_NORMAL) {
                    // printf("const blob name %s\n", name.c_str());
                    blob->SetFlag(DATA_FLAG_CHANGE_NEVER);

                    auto const_node = ConvertToConstNode(const_blobs[name].get());
                    const_node->set_friendly_name(name);

                    auto foreign_blob = new ForeignBlob(blob);
                    foreign_blob->GetBlobDesc().dims = const_blobs[name].get()->GetBufferDims();
                    foreign_blob->SetForeignTensor(std::make_shared<OpenvinoTensor>(const_node));

                    blob_manager_->ReplaceBlob(name, foreign_blob);
                }
            }
        }
    }

    auto const_layers = net_resource->constant_layers;

    for (auto layer_info : net_structure->layers) {
        if (runtime_model_ == RUNTIME_MODE_NORMAL && const_layers.find(layer_info->name) != const_layers.end()) {
            continue;
        }

        LayerType type       = layer_info->type;
        OpenVINOLayerBuilder *cur_layer = CreateOpenVINOLayerBuilder(type);
        
        if (cur_layer == NULL) {
            LOGE("Error: CreateLayerBuilder failed, type:%d\n", type);
            return Status(TNNERR_PARAM_ERR, "CreateLayerBuilder failed");
        }
        std::string layer_name = layer_info->name;
        cur_layer->SetLayerName(layer_name);
        // set layer nodes
        std::vector<Blob *> inputs;
        std::vector<std::string> &input_names = layer_info->inputs;
        // get input nodes
        for (auto name : input_names) {
            ForeignBlob* blob = dynamic_cast<ForeignBlob*>(blob_manager_->GetBlob(name));
            inputs.push_back(blob);
        }
        std::vector<Blob *> outputs;
        std::vector<std::string> &output_names = layer_info->outputs;

#ifdef GENERATE_RESOURCE
        // generate resource if null
        if (net_resource->resource_map.count(layer_name) == 0) {
            LayerParam *layer_param  = layer_info->param.get();
            LayerResource *layer_res = nullptr;
            GenerateRandomResource(type, layer_param, &layer_res, inputs, &net_resource->constant_map);
            net_resource->resource_map[layer_name] = std::shared_ptr<LayerResource>(layer_res);
        }

        std::vector<Blob *> outputs_for_shape;
        for (auto name : output_names) {
            outputs_for_shape.push_back(blob_manager_->GetBlob(name));
        }
        cur_layer->InferShapeAhead(inputs, outputs_for_shape, layer_info->param.get(),
                                   net_resource->resource_map[layer_name].get());
#endif
        // init output nodes
        for (auto name : output_names) {
            auto blob = blob_manager_->GetBlob(name);
            auto foreign_blob = new ForeignBlob(blob);
            foreign_blob->SetForeignTensor(std::make_shared<OpenvinoTensor>());
            blob_manager_->ReplaceBlob(name, foreign_blob);
            blob = foreign_blob;
            outputs.push_back(blob);
        }

        LayerResource *layer_resource = nullptr;
        auto resouce_it = net_resource->resource_map.find(layer_name);
        if (resouce_it != net_resource->resource_map.end()) {
            layer_resource = resouce_it->second.get();
        }

        cur_layer->SetRuntimeMode(runtime_model_);
        cur_layer->SetConstantResource(&net_resource->constant_map);
        cur_layer->SetConstantResourceFlag(&net_resource->constant_blob_flags);
        // init node
        ret = cur_layer->Init(context_, layer_info->param.get(), layer_resource, inputs, outputs, device_);
        if (ret != TNN_OK) {
            LOGE("Error Init layer %s (err: %d or 0x%X)\n", cur_layer->GetLayerName().c_str(), (int)ret, (int)ret);
            return ret;
        }
        
        layers_.push_back(cur_layer);
    }
    return ret;
}

Status OpenVINONetwork_::DeInit() {
    for(auto item : input_blob_map_) {
        delete item.second;
    }
    input_blob_map_.clear();
    for(auto item : output_blob_map_) {
        delete item.second;
    }
    output_blob_map_.clear();
    return TNN_OK;
}

Status OpenVINONetwork_::GetCommandQueue(void **command_queue) {
    return TNN_OK;
}

Status OpenVINONetwork_::Forward() {
    infer_request_.infer();
#if TNN_PROFILE
    std::vector<ov::ProfilingInfo> profiling_info = infer_request_.get_profiling_info();
    for (auto iter : profiling_info) {
	if (std::string(iter.node_type).find("Custom") != std::string::npos) {
	    continue;
	}
        auto pdata = std::make_shared<ProfilingData>();
	pdata->layer_name = iter.node_name;
	pdata->op_name = iter.node_type;
	pdata->kernel_time = iter.cpu_time.count() / 1000.0f;
        context_->AddProfilingData(pdata);
    }
#endif
    return TNN_OK;
}

// @brief openvino instance network infer, it will not wait
Status OpenVINONetwork_::ForwardAsync(Callback call_back) {
    return Forward();
}

Status OpenVINONetwork_::SetCpuNumThreads(int num_threads) {
    std::map<std::string, ov::Any> config = {
        {CONFIG_KEY(CPU_THREADS_NUM), ToString(num_threads)},
        {CONFIG_KEY(CPU_THROUGHPUT_STREAMS), "0"},
        {CONFIG_KEY(CPU_BIND_THREAD), "NO"},
    };

    ie_.set_property("CPU", config);


    BlobMap input_blobs;
    blob_manager_->GetAllInputBlobs(input_blobs);
    InputShapesMap network_shapes;
    for (auto &iter : input_blobs) {
        network_shapes[iter.first] = iter.second->GetBlobDesc().dims;
    }

    // load network again
    return Reshape(network_shapes);
}

}  // namespace TNN_NS
