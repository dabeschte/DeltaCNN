#include "common.cuh"

__device__ DCMetrics *d_metrics;

__global__ void check_d_metrics_ptr() {
    if (threadIdx.x == 0) {
        printf("common.cu &d_metrics=%p\n", d_metrics);
        printf("common.cu &d_metrics.vals_read_dense = %p\n", &d_metrics->n_vals_read_dense);
        printf("common.cu &d_metrics.vals_written_dense = %p\n", &d_metrics->n_vals_written_dense);
        d_metrics->n_active_flops += 1;
        printf("common.cu d_metrics.n_active_flops=%i\n", d_metrics->n_active_flops);
    }
}


bool init_performance_metrics() {
#ifdef ENABLE_METRICS
    const bool use_const_memory = false;
    if (use_const_memory) {
        // uint64_t *d_buffers;
        // HANDLE_ERROR(cudaMalloc(&d_buffers, sizeof(uint64_t) * DCMetrics::n_samples_total));
        // HANDLE_ERROR(cudaMemset(d_buffers, 0, sizeof(uint64_t) * DCMetrics::n_samples_total));
        // h_metrics.n_active_tiles = &d_buffers[0];
        // h_metrics.n_tiles = &d_buffers[1];
        // h_metrics.n_active_inputs = &d_buffers[2];
        // h_metrics.n_inputs = &d_buffers[3];
        // h_metrics.n_tiles_sparse_mode = &d_buffers[4];
        // h_metrics.n_tiles_dense_mode = &d_buffers[5];
        // h_metrics.n_active_flops = &d_buffers[6];
        // h_metrics.n_theoretical_flops = &d_buffers[7];
        // h_metrics.n_dense_flops = &d_buffers[8];
        // h_metrics.n_vals_read = &d_buffers[9];
        // h_metrics.n_vals_read_dense = &d_buffers[10];
        // h_metrics.n_vals_written = &d_buffers[11];
        // h_metrics.n_vals_written_dense = &d_buffers[12];
        // h_metrics.active_input_histogram = &d_buffers[13];
        // printf("host address of v_vals_read_dense =%p\n", h_metrics.n_vals_read_dense);
        // printf("host address of v_vals_written_dense =%p\n", h_metrics.n_vals_written_dense);
        // copy_performance_metrics_to_gpu();
    } else {
        HANDLE_ERROR(cudaMalloc(&d_metrics_ptr_copy, sizeof(DCMetrics)));
        HANDLE_ERROR(cudaMemset(d_metrics_ptr_copy, 0, sizeof(DCMetrics)));
        copy_performance_metrics_to_gpu(d_metrics);
    }
    return true;
#else
    return false;
#endif
}

void copy_performance_metrics_to_gpu(DCMetrics*& d) {
#ifdef ENABLE_METRICS
    // const bool use_const_memory = false;
    // if (use_const_memory) {
    //     HANDLE_ERROR(cudaMemcpyToSymbol(d_metrics, &h_metrics, sizeof(DCMetrics)));
    // }
    HANDLE_ERROR(cudaMemcpyToSymbol(d, &d_metrics_ptr_copy, sizeof(DCMetrics*)));
    // DCMetrics *d_ptr_addr;
    // HANDLE_ERROR(cudaGetSymbolAddress((void**) &d_ptr_addr, d));
    // printf("common.cu d_metrics_address=%p\n", d_ptr_addr);
    // check_d_metrics_ptr<<<1,32>>>();
#endif
}

void reset_performance_metrics() {
#ifdef ENABLE_METRICS
    // const int use_const_memory = false;
    // if (use_const_memory) {
    //     HANDLE_ERROR(cudaMemset(h_metrics.n_active_tiles, 0, sizeof(uint64_t) * DCMetrics::n_samples_total));  
    // } else {
    //     HANDLE_ERROR(cudaMemset(d_metrics, 0, sizeof(DCMetrics)));
    // }
    HANDLE_ERROR(cudaMemset(d_metrics_ptr_copy, 0, sizeof(DCMetrics)));
#endif
}

std::vector<torch::Tensor> retrieve_metrics() {
#ifdef ENABLE_METRICS
    // int64_t *h_buffers = (int64_t*) malloc(sizeof(int64_t) * DCMetrics::n_samples_total);
    // HANDLE_ERROR(cudaMemcpy(h_buffers, d_metrics_ptr_copy, sizeof(DCMetrics), cudaMemcpyDeviceToHost));
    
    DCMetrics h_d_metric;
    HANDLE_ERROR(cudaMemcpy(&h_d_metric, d_metrics_ptr_copy, sizeof(DCMetrics), cudaMemcpyDeviceToHost));
    
    torch::Tensor tiles = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor inputs = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor mode = torch::zeros({2}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor flops = torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor memtransfer = torch::zeros({4}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor histrogram = torch::zeros({DCMetrics::histogram_samples}, torch::TensorOptions().dtype(torch::kInt64));

    // tiles.data_ptr<int64_t>()[0] = int64_t(h_buffers[0]);
    // tiles.data_ptr<int64_t>()[1] = int64_t(h_buffers[1]);
    // inputs.data_ptr<int64_t>()[0] = int64_t(h_buffers[2]);
    // inputs.data_ptr<int64_t>()[1] = int64_t(h_buffers[3]);
    // mode.data_ptr<int64_t>()[0] = int64_t(h_buffers[4]);
    // mode.data_ptr<int64_t>()[1] = int64_t(h_buffers[5]);
    // flops.data_ptr<int64_t>()[0] = int64_t(h_buffers[6]);
    // flops.data_ptr<int64_t>()[1] = int64_t(h_buffers[7]);
    // flops.data_ptr<int64_t>()[2] = int64_t(h_buffers[8]);
    // memtransfer.data_ptr<int64_t>()[0] = int64_t(h_buffers[9]);
    // memtransfer.data_ptr<int64_t>()[1] = int64_t(h_buffers[10]);
    // memtransfer.data_ptr<int64_t>()[2] = int64_t(h_buffers[11]);
    // memtransfer.data_ptr<int64_t>()[3] = int64_t(h_buffers[12]);

    tiles.data_ptr<int64_t>()[0] = int64_t(h_d_metric.n_active_tiles);
    tiles.data_ptr<int64_t>()[1] = int64_t(h_d_metric.n_tiles);
    inputs.data_ptr<int64_t>()[0] = int64_t(h_d_metric.n_active_inputs);
    inputs.data_ptr<int64_t>()[1] = int64_t(h_d_metric.n_inputs);
    mode.data_ptr<int64_t>()[0] = int64_t(h_d_metric.n_tiles_sparse_mode);
    mode.data_ptr<int64_t>()[1] = int64_t(h_d_metric.n_tiles_dense_mode);
    flops.data_ptr<int64_t>()[0] = int64_t(h_d_metric.n_active_flops);
    flops.data_ptr<int64_t>()[1] = int64_t(h_d_metric.n_theoretical_flops);
    flops.data_ptr<int64_t>()[2] = int64_t(h_d_metric.n_dense_flops);
    memtransfer.data_ptr<int64_t>()[0] = int64_t(h_d_metric.n_vals_read);
    memtransfer.data_ptr<int64_t>()[1] = int64_t(h_d_metric.n_vals_read_dense);
    memtransfer.data_ptr<int64_t>()[2] = int64_t(h_d_metric.n_vals_written);
    memtransfer.data_ptr<int64_t>()[3] = int64_t(h_d_metric.n_vals_written_dense);

    int64_t *histogram_ptr = histrogram.data_ptr<int64_t>();
    for (int i = 0; i < DCMetrics::histogram_samples; i++) {
        // histogram_ptr[i] = int64_t(h_buffers[i+(DCMetrics::n_samples_total-DCMetrics::histogram_samples)]);
        histogram_ptr[i] = int64_t(h_d_metric.active_input_histogram[i]);
    }
    return {tiles, inputs, mode, flops, memtransfer, histrogram};
#endif
    return {};
}