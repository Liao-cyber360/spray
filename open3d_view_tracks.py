import json
import open3d as o3d

def visualize_tracks(jsonl_file):
    # Load points from the JSONL file
    points_by_camera = {'left': [], 'right': []}
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            camera = data['camera']
            points = data['points']
            points_by_camera[camera].extend(points)

    # Create PointCloud objects for each camera
    pcd_left = o3d.geometry.PointCloud()
    pcd_left.points = o3d.utility.Vector3dVector(points_by_camera['left'])
    pcd_right = o3d.geometry.PointCloud()
    pcd_right.points = o3d.utility.Vector3dVector(points_by_camera['right'])

    # Optionally visualize using LineSet or PointClouds
    line_set = o3d.geometry.LineSet()
    # Create lines and color them based on time (to be implemented)

    # Visualize
    o3d.visualization.draw_geometries([pcd_left, pcd_right])

if __name__ == '__main__':' 
    visualize_tracks('OUTPUT_JSONL')
