# Sign detection from point cloud

    Sign detection and classification by point cloud.
    Algorithm is tested on simuatied data, where plate is highly reflective.

### Provided materials:

 - Python files:
    - main.py
    - detectors.py
    - io_utils.py
    - lin_alg.py
 - Jupiter notebook solution.ipynb
 - Data files in "Objects" directory:
   - Simulated data:
      - Directory rot_x
      - Directory rot_z
      - Directory straight
      - Each directory contains four files: pole.ply, square.ply, triangle.ply, circle.ply
   
### Instructions:

   1. Set python > 3.8 environment according to requirements.txt
   2. To test algorithm on the provided data run main.py either from terminal or IDE with default settings.
   3. To run tests on provided and simulated data there is also a Jupyter notebook.
      1. It is already pre runned, therefore results will show up automatically.
      2. You may additionally re-run it as specified within the notebook.
