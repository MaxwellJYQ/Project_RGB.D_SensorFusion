### Project Name
**Project_RGB_D_SensorFusion**

### Project Description
This project implements a model for Guided Depth Map Super-Resolution (GDSR) using a Discrete Cosine Transform Network (DCTNet). The project includes the model implementation, training framework, and testing framework. It is primarily inspired by the paper "Discrete Cosine Transform Network for Guided Depth Map Super-Resolution."

### File Structure
- `model.py`: Model implementation
- `train.py`: Training script
- `test.py`: Testing script
- `pre_process.py`: Preprocessing script to generate blurred images
- `Demonstration of pre-processing effect.pdf`: Demonstration of pre-processing effect
- `Demonstration of project results.pdf`: Demonstration of project results
- `requirements.txt`: Project dependencies
- `utils_gdsr.py`: Utility functions
- `dct.py`: Discrete Cosine Transform related code
- `demo.py`: Demonstration script
- `metrics.py`: Evaluation metrics
- `processing_testsets.py`: Test set processing script

### Dataset Information
Due to company policy, the dataset used in this project is internal and cannot be made public. Please prepare a similar dataset for testing purposes.

### Installation and Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Project_RGB_D_SensorFusion.git
   cd Project_RGB_D_SensorFusion
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**:
   ```bash
   python train.py
   ```

4. **Test the model**:
   ```bash
   python test.py
   ```

### Citation
If you use this project in your research, please cite the following paper:
- Title: Discrete Cosine Transform Network for Guided Depth Map Super-Resolution
- Authors: Zixiang Zhao, Jiangshe Zhang, Shuang Xu, Zudi Lin, Hanspeter Pfister
- Link: [arXiv:2104.06977](https://arxiv.org/abs/2104.06977)

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments
Thanks to Zixiang Zhao and co-authors for their research and code contributions in the paper "Discrete Cosine Transform Network for Guided Depth Map Super-Resolution."
