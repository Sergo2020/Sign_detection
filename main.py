from pathlib import Path
import mesh_utils as ply

if __name__ == '__main__':
    obj_path = Path(r"C:\Users\Serge\Desktop\Innoviz\Objects\cube.ply")
    clster_path = Path(r"C:\Users\Serge\Desktop\Innoviz\Objects\cluster_2.csv")


    points = ply.read_csv(clster_path)
    ply.show_mesh(points)