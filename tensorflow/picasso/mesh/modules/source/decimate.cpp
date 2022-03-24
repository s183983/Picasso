#include <cuda/cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include <iostream>
using namespace tensorflow;


REGISTER_OP("MeshDecimation")
  .Attr("use_area: bool")            // indicates if use area to weigh Quadric errors
  .Attr("wgt_bnd: float")            // penalty weight for boundary edges
  .Input("vertex_input: float32")    // concat_Nv * Dim(Dim>=3 at least xyz)
  .Input("face_input: int32")        // concat_Nf * 3
  .Input("normal_input: float32")    // concat_Nf * 5, [normal,d,area] of faces
  .Input("nv_input: int32")          // batch: each element is the vertex number of an input sample
  .Input("mf_input: int32")          // batch: each element is the face   number of an input sample
  .Input("nv_remove: int32")         // batch: expected number of vertices to remove in an input sample
  .Output("vertex_output: float32")  // concat_Nv * Dim(Dim>=3 at least xyz)
  .Output("face_output: int32")      // concat_Nf * 3
  .Output("is_degenerate: bool")     // concat_Nf: (if the contracted face is degenerate/silver triangle or not)
  .Output("vertex_replace: int32")   // concat_Nv: (vertex replacement: clustering information)
  .Output("vertex_map: int32")       // concat_Nv: (vertex mapping: map input to output vertices)
  .Output("nv_output: int32")        // batch: each element is the vertex number of an output sample
  .Output("mf_output: int32")        // batch: each element is the face   number of an output sample
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    // input shape getting and checking
    ::tensorflow::shape_inference::ShapeHandle vertex_input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &vertex_input));
    ::tensorflow::shape_inference::ShapeHandle face_input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &face_input));
    ::tensorflow::shape_inference::ShapeHandle normal_input;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &normal_input));

    // output shape setting
    c->set_output(0, c->input(0));  // vertexOut <- vertexIn
    c->set_output(1, c->input(1));  // faceOut   <- faceIn
    ::tensorflow::shape_inference::ShapeHandle is_degenerate = c->MakeShape({c->Dim(face_input, 0)});
    c->set_output(2, is_degenerate);
    ::tensorflow::shape_inference::ShapeHandle vertex_replace = c->MakeShape({c->Dim(vertex_input, 0)});
    c->set_output(3, vertex_replace);
    c->set_output(4, vertex_replace);

    c->set_output(5, c->input(3));
    c->set_output(6, c->input(4));

    return Status::OK();
  });
REGISTER_OP("CombineClusters")
  .Input("rep_a: int32")      // concat_Nv: (vertex replacement: clustering information)
  .Input("map_a: int32")      // concat_Nv: (vertex mapping: map input to output vertices)
  .Input("rep_b: int32")      // concat_Nv: (vertex replacement: clustering information)
  .Input("map_b: int32")      // concat_Nv: (vertex mapping: map input to output vertices)
  .Output("rep_out: int32")   // concat_Nv: (vertex replacement: clustering information)
  .Output("map_out: int32")   // concat_Nv: (vertex mapping: map input to output vertices)
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    // output shape setting
    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));
    return Status::OK();
  });
REGISTER_OP("CountVertexAdjface")
  .Input("face: int32")         // face vertex list: concat_NfIn * 3
  .Input("vt_map: int32")       // vertex mapping from input to output vertices: concat_NvIn
  .Input("vt_out: float32")     // vertices in decimated mesh: concat_NvOut * 3
  .Output("nf_count: int32")    // the number of adjacency faces of each output vertex: concat_NvOut
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle vertex_out;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &vertex_out));

    // output shape setting
    ::tensorflow::shape_inference::ShapeHandle nf_count = c->MakeShape({c->Dim(vertex_out, 0)});
    c->set_output(0, nf_count);
    return Status::OK();
  });


void meshDecimationLauncher(const bool useArea, const float wgtBnd,       //hyperparams
                            const int B, const int D, const int Nv, const int Nf, const int* nvIn, const int* mfIn,  //inputs
                            const int* nv2Remove, const float* vertexIn, const int* faceIn, const float* planeIn,    //inputs
                            int* nvOut, int* mfOut, float* vertexOut, int* faceOut, int* vtReplace, int* vtMap,      //ouputs
                            bool* isDegenerate);
class MeshDecimationGpuOp: public OpKernel{
    public:
        explicit MeshDecimationGpuOp(OpKernelConstruction* context):OpKernel(context)
        {
            OP_REQUIRES_OK(context, context->GetAttr("use_area", &use_area_));
            OP_REQUIRES_OK(context, context->GetAttr("wgt_bnd",  &wgt_bnd_));
            OP_REQUIRES(context, wgt_bnd_ >= 0, errors::InvalidArgument("Mesh decimation requires wgt_bnd>=0, got ", wgt_bnd_));
        }
        void Compute(OpKernelContext * context) override {
            // Grab the input tensors
            const Tensor& vertexIn_tensor  = context->input(0);
            const Tensor& faceIn_tensor    = context->input(1);
            const Tensor& planeIn_tensor   = context->input(2);
            const Tensor& nvIn_tensor      = context->input(3);
            const Tensor& mfIn_tensor      = context->input(4);
            const Tensor& nv2Remove_tensor = context->input(5);

            // get the dims required by computations
            int Nv = vertexIn_tensor.shape().dim_size(0);  // number of input vertices/points
            int D  = vertexIn_tensor.shape().dim_size(1);  // dimension of input vertices
            int Nf = faceIn_tensor.shape().dim_size(0);    // number of input faces
            int B  = nvIn_tensor.shape().dim_size(0);      // batch size

//            std::cout<<"The input vertex dimension is "<<D<<std::endl;

            // conditional checks and validation
            OP_REQUIRES(context, vertexIn_tensor.dims()==2 && vertexIn_tensor.shape().dim_size(1)>=3,
                                 errors::InvalidArgument("The shape of input vertex should be (Nv, 3+))."));
            OP_REQUIRES(context, faceIn_tensor.dims()==2 && faceIn_tensor.shape().dim_size(1)==3,
                                 errors::InvalidArgument("The shape of input face should be (Nf, 3))."));
            OP_REQUIRES(context, planeIn_tensor.dims()==2 && planeIn_tensor.shape().dim_size(1)==5,
                                 errors::InvalidArgument("The shape of input face normals should be (Nf, 5))."));
            OP_REQUIRES(context, Nf==planeIn_tensor.shape().dim_size(0),
                                 errors::InvalidArgument("The shape of input face and normal should be identical."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(nvIn_tensor.shape()),
                errors::InvalidArgument("nvIn expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(mfIn_tensor.shape()),
                errors::InvalidArgument("mfIn expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(nv2Remove_tensor.shape()),
                errors::InvalidArgument("nv2Remove expects a 1-D vector."));

            // flatten the input tensors
            auto vertexIn_flat = vertexIn_tensor.flat<float>();
            auto faceIn_flat   = faceIn_tensor.flat<int32>();
            auto planeIn_flat  = planeIn_tensor.flat<float>();
            auto nvIn_flat     = nvIn_tensor.flat<int32>();
            auto mfIn_flat     = mfIn_tensor.flat<int32>();
            auto nv2Remove_flat    = nv2Remove_tensor.flat<int32>();

            // Create an output tensor
            Tensor* vertexOut_tensor    = NULL;
            Tensor* faceOut_tensor      = NULL;
            Tensor* isDegenerate_tensor = NULL;
            Tensor* vtReplace_tensor    = NULL;
            Tensor* vtMap_tensor        = NULL;
            Tensor* nvOut_tensor        = NULL;
            Tensor* mfOut_tensor        = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{Nv,D}, &vertexOut_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{Nf,3}, &faceOut_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape{Nf}, &isDegenerate_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(3, TensorShape{Nv}, &vtReplace_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(4, TensorShape{Nv}, &vtMap_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(5, TensorShape{B}, &nvOut_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(6, TensorShape{B}, &mfOut_tensor));
            auto vertexOut_flat    = vertexOut_tensor->flat<float>();
            auto faceOut_flat      = faceOut_tensor->flat<int32>();
            auto isDegenerate_flat = isDegenerate_tensor->flat<bool>();
            auto vtReplace_flat    = vtReplace_tensor->flat<int32>();
            auto vtMap_flat        = vtMap_tensor->flat<int32>();
            auto nvOut_flat        = nvOut_tensor->flat<int32>();
            auto mfOut_flat        = mfOut_tensor->flat<int32>();

            const float* vertexIn  = &(vertexIn_flat(0));
            const int*   faceIn    = &(faceIn_flat(0));
            const float* planeIn   = &(planeIn_flat(0));
            const int*   nvIn      = &(nvIn_flat(0));
            const int*   mfIn      = &(mfIn_flat(0));
            const int*   nv2Remove = &(nv2Remove_flat(0));

            float* vertexOut   = &(vertexOut_flat(0));
            int*  faceOut      = &(faceOut_flat(0));
            bool* isDegenerate = &(isDegenerate_flat(0));
            int*  vtReplace    = &(vtReplace_flat(0));
            int*  vtMap        = &(vtMap_flat(0));
            int*  nvOut        = &(nvOut_flat(0));
            int*  mfOut        = &(mfOut_flat(0));
            cudaMemset(vertexOut,0,sizeof(float)*Nv*D);
            cudaMemset(faceOut,0,sizeof(int)*Nf*3);
            cudaMemset(isDegenerate,false,sizeof(bool)*Nf);
            cudaMemset(nvOut,0,sizeof(int)*B);
            cudaMemset(mfOut,0,sizeof(int)*B);
            meshDecimationLauncher(use_area_, wgt_bnd_,
                                   B, D, Nv, Nf, nvIn, mfIn, nv2Remove, vertexIn, faceIn, planeIn,
                                   nvOut, mfOut, vertexOut, faceOut, vtReplace, vtMap, isDegenerate);
        }
    private:
        bool use_area_;
        float wgt_bnd_;
};
REGISTER_KERNEL_BUILDER(Name("MeshDecimation").Device(DEVICE_GPU),MeshDecimationGpuOp);


void combineClustersLauncher(const int nvA, const int nvB, const int* repA, const int* mapA,
                             const int* repB, const int* mapB, int* repOut, int* mapOut);
class CombineClustersGpuOp: public OpKernel{
    public:
        explicit CombineClustersGpuOp(OpKernelConstruction* context):OpKernel(context) {}

        void Compute(OpKernelContext * context) override {
            // Grab the input tensors
            const Tensor& repA_tensor = context->input(0);
            const Tensor& mapA_tensor = context->input(1);
            const Tensor& repB_tensor = context->input(2);
            const Tensor& mapB_tensor = context->input(3);

            // get the dims required by computations
            int nvA = repA_tensor.shape().dim_size(0);    // number of input faces
            int nvB = repB_tensor.shape().dim_size(0); // number of input vertices/points

//            std::cout<<"repA_n, mapA_n: "<<nvA<<", "<<mapA_tensor.shape().dim_size(0)<<std::endl;
//            std::cout<<"repB_n, mapB_n: "<<nvB<<", "<<mapB_tensor.shape().dim_size(0)<<std::endl;

            // conditional checks and validation
            OP_REQUIRES(context, repA_tensor.shape().dim_size(0)==mapA_tensor.shape().dim_size(0),
                        errors::InvalidArgument("The shape of repA and mapA should be identical."));
            OP_REQUIRES(context, repB_tensor.shape().dim_size(0)==mapB_tensor.shape().dim_size(0),
                        errors::InvalidArgument("The shape of repB and mapB should be identical."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(repA_tensor.shape()),
                errors::InvalidArgument("repA expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(mapA_tensor.shape()),
                errors::InvalidArgument("mapA expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(repB_tensor.shape()),
                errors::InvalidArgument("repB expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(mapB_tensor.shape()),
                errors::InvalidArgument("mapB expects a 1-D vector."));

            // flatten the input tensors
            auto repA_flat = repA_tensor.flat<int32>();
            auto mapA_flat = mapA_tensor.flat<int32>();
            auto repB_flat = repB_tensor.flat<int32>();
            auto mapB_flat = mapB_tensor.flat<int32>();

            // Create an output tensor
            Tensor* repOut_tensor = NULL;
            Tensor* mapOut_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{nvA}, &repOut_tensor));
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{nvA}, &mapOut_tensor));
            auto repOut_flat = repOut_tensor->flat<int32>();
            auto mapOut_flat = mapOut_tensor->flat<int32>();

            const int* repA = &(repA_flat(0));
            const int* mapA = &(mapA_flat(0));
            const int* repB = &(repB_flat(0));
            const int* mapB = &(mapB_flat(0));

            int* repOut = &(repOut_flat(0));
            int* mapOut = &(mapOut_flat(0));
            combineClustersLauncher(nvA, nvB, repA, mapA, repB, mapB, repOut, mapOut);
        }
};
REGISTER_KERNEL_BUILDER(Name("CombineClusters").Device(DEVICE_GPU),CombineClustersGpuOp);


void countVertexAdjfaceLauncher(int NfIn, const int* face, const int* vtMap, int* nfcount);
class CountVertexAdjfaceGpuOp: public OpKernel{
    public:
        explicit CountVertexAdjfaceGpuOp(OpKernelConstruction* context):OpKernel(context) {}

        void Compute(OpKernelContext * context) override {
            // Grab the input tensors
            const Tensor& faceIn_tensor = context->input(0);
            const Tensor& vtMap_tensor  = context->input(1);
            const Tensor& vtOut_tensor  = context->input(2);

            // get the dims required by computations
            int NfIn  = faceIn_tensor.shape().dim_size(0);    // number of input faces
            int NvOut = vtOut_tensor.shape().dim_size(0);   // number of output vertices/points

            // conditional checks and validation
            OP_REQUIRES(context, faceIn_tensor.dims()==2 && faceIn_tensor.shape().dim_size(1)==3,
                                 errors::InvalidArgument("The shape of input face should be (NfIn, 3))."));
            OP_REQUIRES(context, vtOut_tensor.dims()==2 && vtOut_tensor.shape().dim_size(1)==3,
                        errors::InvalidArgument("The shape of vertexOut should be (NvOut,3)."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vtMap_tensor.shape()),
                errors::InvalidArgument("vtMap expects a 1-D vector."));

            // flatten the input tensors
            auto faceIn_flat  = faceIn_tensor.flat<int32>();
            auto vtMap_flat = vtMap_tensor.flat<int32>();

            // Create an output tensor
            Tensor* nfCount_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{NvOut}, &nfCount_tensor));
            auto nfCount_flat = nfCount_tensor->flat<int32>();

            const int* faceIn = &(faceIn_flat(0));
            const int* vtMap  = &(vtMap_flat(0));

            int* nfCount = &(nfCount_flat(0));
            cudaMemset(nfCount,0,sizeof(int)*NvOut);  // initialize nfCount all to zeros
            countVertexAdjfaceLauncher(NfIn, faceIn, vtMap, nfCount);
        }
};
REGISTER_KERNEL_BUILDER(Name("CountVertexAdjface").Device(DEVICE_GPU),CountVertexAdjfaceGpuOp);






