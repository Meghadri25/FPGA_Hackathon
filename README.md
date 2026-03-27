# FPGA Neural Network for Energy Forecasting

This project implements a fixed-point neural network on PYNQ Z2 FPGA for multi-client energy demand forecasting.

## Project Structure

```
FPGA_Hackathon/
├── src/                    # Python source code
│   ├── train_decision_engine.py    # Training script
│   ├── fixed_point_model.py        # Fixed-point evaluation
│   ├── process_all_models.py       # Model processing
│   ├── convert_to_fixed_point.py   # Fixed-point conversion
│   ├── generate_coe.py            # Generate BRAM coefficients
│   ├── generate_verilog.py        # Generate individual Verilog
│   ├── get_test_data.py           # Generate test vectors
│   └── ...                        # Other utility scripts
├── verilog/               # SystemVerilog HDL files
│   ├── nn_combined.sv     # Combined module for all clients
│   └── nn_client_*.sv     # Individual client modules
├── test/                  # Test benches
│   └── nn_tb.sv           # Test bench for verification
├── bram/                  # BRAM initialization files
│   ├── weights_*.coe      # Vivado coefficient files
│   └── weights_*.hex      # Simulation hex files
├── data/                  # Datasets and predictions
│   ├── LD2011_2014.txt    # Main dataset
│   ├── continuous dataset.csv
│   └── *_predictions.csv  # Prediction outputs
├── weights/               # Model weights and parameters
│   ├── model_MT_*.pt      # PyTorch models
│   ├── model_params_*.npz # Extracted parameters
│   ├── fixed_params_*.npz # Fixed-point parameters
│   └── quantized_params.npz
└── README.md
```

## Usage

1. **Training**: Run `src/train_decision_engine.py` to train models
2. **Processing**: Use `src/process_all_models.py` to extract weights
3. **Fixed-Point**: Run `src/convert_to_fixed_point.py` for quantization
4. **FPGA Code**: `src/generate_coe.py` and `src/generate_verilog.py`
5. **Simulation**: Use `test/nn_tb.sv` in Vivado/ModelSim
6. **Synthesis**: Load `verilog/nn_combined.sv` and `bram/weights_*.coe` in Vivado

## FPGA Implementation

- **Target**: PYNQ Z2 (Zynq-7000)
- **Format**: 16-bit fixed-point (Q8.8)
- **DSP Slices**: Optimized for DSP48 usage
- **BRAM**: 5 blocks for client-specific weights
- **Architecture**: 3-layer MLP (24→128→64→32) + 2 heads

## Clients

- MT_196, MT_279, MT_362, MT_364, MT_370

Each with separate weights for personalized forecasting.