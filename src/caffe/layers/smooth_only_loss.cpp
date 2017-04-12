#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/smooth_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SmoothOnlyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}

template <typename Dtype>
void SmoothOnlyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
Dtype SmoothOnlyLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SmoothOnlyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int spatial_dim = prob_.height() * prob_.width();
  int H = prob_.height();
  int W = prob_.width();
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }

  //LOG(INFO) << "loss from softmax loss: " << loss; 
  // pairwise terms 
  double diff_t = 0;
  double diff_b = 0;
  double diff_l = 0;
  double diff_r = 0;
  for (int i=0; i<outer_num_; ++i){
    for (int j=0; j<spatial_dim; j++){
      const int label_value = static_cast<int>(label[i * spatial_dim + j]); // y_i
      double Py_i = prob_data[i*dim + label_value*spatial_dim + j]; //p_ground truth
      //LOG(INFO) << "current pixel label, proba: " << label_value << " , " << Py_i; 
      int count_sim_neighbors = 0; 
      // neighbors labels
      if (j >= H) {
        const int y_top = static_cast<int>(label[i * spatial_dim + j-H]);
        double Py_top = prob_data[i*dim + y_top*spatial_dim + j-H]; 
        if ( label_value == y_top  )  { diff_t = abs(Py_i - Py_top);   count_sim_neighbors++; } 
        //LOG(INFO) << " top: " << y_top << " , " << Py_top; 
      }
      if (j+H <= spatial_dim) {
        const int y_bottom = static_cast<int>(label[i * spatial_dim + j+H]); 
        double Py_bottom = prob_data[i*dim + y_bottom*spatial_dim + j+H]; 
        if ( label_value == y_bottom) { diff_b = abs(Py_i - Py_bottom); count_sim_neighbors++;}
        //LOG(INFO) << " bottom: " << y_bottom << " , " << Py_bottom;
      }
      if ( (j% W ) != 0 ) {
        const int y_left = static_cast<int>(label[i * spatial_dim + j-1]); 
        double Py_left = prob_data[i*dim + y_left*spatial_dim + j-1]; 
        if ( label_value == y_left  ) { diff_l = abs(Py_i - Py_left);   count_sim_neighbors++;}
        //LOG(INFO) << " left: " << y_left << " , " << Py_left;
      }
      if ( ( (j+1) % W ) != 0 ) {
        const int y_right = static_cast<int>(label[i * spatial_dim + j+1]); 
        double Py_right = prob_data[i*dim + y_right*spatial_dim + j+1]; 
        if ( label_value == y_right ) { diff_r = abs(Py_i - Py_right);  count_sim_neighbors++;}
        //LOG(INFO) << " right: " << y_right << " , " << Py_right;
      } 
      // add to the loss the differences only if the 4 connected neighbors share labels
      //LOG(INFO) << "diffs: " << diff_t << " " << diff_b << " " << diff_l << " " << diff_r; 
      //LOG(INFO) << "sim nei: " << count_sim_neighbors; 
      if (count_sim_neighbors == 4) { loss += diff_t + diff_b + diff_l + diff_r;}     
    }
  }  
     
  //LOG(INFO) << "loss after pairwise loss: " << loss; 



  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SmoothOnlyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();
    int H = prob_.height();
    int W = prob_.width(); 
    int count = 0;
    // gradient per pixel
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        const int label_value = static_cast<int>(label[i * spatial_dim + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->channels(); ++c) {
            bottom_diff[i * dim + c * spatial_dim + j] = 0;
          }
        } 
        else {
          // gradient softmax
          bottom_diff[i * dim + label_value * spatial_dim + j] -= 1;
          
          // gradient pairwise terms 
          int count_sim_neighbors = 0; 
          double Py_i = prob_data[i*dim + label_value*spatial_dim + j]; //p_ground truth
          double Py_top = 0;
      double Py_bottom = 0; 
          double Py_left = 0;
          double Py_right = 0; 
          // top neighbor
          if (j >= H) {
     const int y_top = static_cast<int>(label[i * spatial_dim + j-H]);
     Py_top = prob_data[i*dim + y_top*spatial_dim + j-H]; 
           if ( (label_value == y_top)  )  { count_sim_neighbors++; } 
          }
          // bottom neighbor
          if (j+H <= spatial_dim) {
     const int y_bottom = static_cast<int>(label[i * spatial_dim + j+H]); 
           Py_bottom = prob_data[i*dim + y_bottom*spatial_dim + j+H]; 
           if ( (label_value == y_bottom)  )  { count_sim_neighbors++;  } 
          }
          // left neighbor
          if ( ( j% W ) != 0 ) {
     const int y_left = static_cast<int>(label[i * spatial_dim + j-1]); 
           Py_left = prob_data[i*dim + y_left*spatial_dim + j-1]; 
           if ( (label_value == y_left)  )  {  count_sim_neighbors++;  } 
          }
          // right neighbor
          if ( ( (j+1) % W ) != 0 ) {
     const int y_right = static_cast<int>(label[i * spatial_dim + j+1]); 
           Py_right = prob_data[i*dim + y_right*spatial_dim + j+1]; 
           if ( (label_value == y_right)  )  {count_sim_neighbors++;  } 
          } 
          // if the 4 connected neighbors share label: 
          if (count_sim_neighbors==4) {
             // y_top
       if (Py_i >= Py_top) { bottom_diff[i * dim + label_value * spatial_dim + j]   +=  (Py_i - Py_top)*(Py_i - Py_i*Py_i); 
                                   bottom_diff[i * dim + label_value * spatial_dim + j-H] +=  (Py_i - Py_top)*(-Py_top + Py_top*Py_top); }
             if (Py_i < Py_top) { bottom_diff[i * dim + label_value * spatial_dim + j]    += -(Py_i - Py_top)*(Py_i - Py_i*Py_i); 
                                   bottom_diff[i * dim + label_value * spatial_dim + j-H] += -(Py_i - Py_top)*(-Py_top + Py_top*Py_top); }  
      // y_bottom
      if (Py_i >= Py_bottom) { bottom_diff[i * dim + label_value * spatial_dim + j]   +=  (Py_i - Py_bottom)*(Py_i - Py_i*Py_i); 
                                     bottom_diff[i * dim + label_value * spatial_dim + j+H] +=  (Py_i - Py_bottom)*(-Py_bottom + Py_bottom*Py_bottom); }
            if (Py_i < Py_bottom) { bottom_diff[i * dim + label_value * spatial_dim + j]    += -(Py_i - Py_bottom)*(Py_i - Py_i*Py_i); 
                                     bottom_diff[i * dim + label_value * spatial_dim + j+H] += -(Py_i - Py_bottom)*(-Py_bottom + Py_bottom*Py_bottom); } 
      // y_left
      if (Py_i >= Py_left) { bottom_diff[i * dim + label_value * spatial_dim + j]   +=  (Py_i - Py_left)*(Py_i - Py_i*Py_i); 
                                   bottom_diff[i * dim + label_value * spatial_dim + j-1] +=  (Py_i - Py_left)*(-Py_left + Py_left*Py_left); }
            if (Py_i < Py_left) { bottom_diff[i * dim + label_value * spatial_dim + j]    += -(Py_i - Py_left)*(Py_i - Py_i*Py_i); 
                                   bottom_diff[i * dim + label_value * spatial_dim + j-1] += -(Py_i - Py_left)*(-Py_left + Py_left*Py_left); } 
      // y_right
      if (Py_i >= Py_right) { bottom_diff[i * dim + label_value * spatial_dim + j]   +=  (Py_i - Py_right)*(Py_i - Py_i*Py_i); 
                                    bottom_diff[i * dim + label_value * spatial_dim + j+1] +=  (Py_i - Py_right)*(-Py_right + Py_right*Py_right); }
            if (Py_i < Py_right) { bottom_diff[i * dim + label_value * spatial_dim + j]    += -(Py_i - Py_right)*(Py_i - Py_i*Py_i); 
                                    bottom_diff[i * dim + label_value * spatial_dim + j+1] += -(Py_i - Py_right)*(-Py_right + Py_right*Py_right); }
        }   
        // update count
        ++count;
        }// close else
      }// close for j
    }// close for i

    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

// #ifdef CPU_ONLY
//STUB_GPU(SmoothOnlyLossLayer);
// #endif

INSTANTIATE_CLASS(SmoothOnlyLossLayer);
REGISTER_LAYER_CLASS(SmoothOnlyLoss);

}  // namespace caffe
