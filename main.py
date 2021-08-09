from pathlib import Path
import mesh_utils as ply
# https://github.com/falcondai/py-ransac/blob/master/ransac.py

if __name__ == '__main__':
    obj_path = Path(r"C:\Users\Serge\Desktop\Innoviz\Objects\cube.ply")
    clster_path = Path(r"C:\Users\Serge\Desktop\Innoviz\Objects\cluster_2.csv")


    points = ply.read_csv(clster_path)
    ply.show_cluster(points)