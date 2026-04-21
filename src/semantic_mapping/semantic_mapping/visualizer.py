import numpy as np
import rerun as rr
import open3d as o3d
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from .utils import generate_colors
from .single_object_new import SingleObject

class VisualizerBase():
    def __init__(self) -> None:
        pass

    def visualize(self, voxel_grid):
        pass

class VisualizerRerun(VisualizerBase):
    def __init__(self) -> None:
        super().__init__()

        rr.init("object_mapping", spawn=True)

        # rr.connect()
        # rr.save("../output/rr_output/rerun.rrd")

        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

        rr.log(
            "world/odom",
            rr.Arrows3D(
                vectors=[[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

        self.vis_percentile_thresh = 0.8
        self.buffer_size = 500
        self.cnt = 0

        self.pca_solver = PCA(n_components=2)

    def visualize_debug(self, object_list, odom, regularized=True):
        camera_to_world = rr.Transform3D(translation=odom['position'], rotation=rr.Quaternion(xyzw=odom['orientation']))
        rr.log("world/odom", camera_to_world)

        colors_list = generate_colors(len(object_list), is_int=False)
        point_colors = [
            (0, 0, 255),      # Deep Blue
            (51, 51, 255),    # Royal Blue
            (102, 102, 255),  # Light Blue
            (153, 153, 255),  # Periwinkle
            (204, 204, 255),  # Lavender
            (255, 204, 255),  # Light Purple
            (255, 153, 204),  # Pink
            (255, 102, 102),  # Light Red
            (255, 51, 51),    # Orange-Red
            (255, 0, 0),      # Bright Red
            (255, 128, 0),    # Deep Orange
            (255, 204, 0),    # Gold
            (255, 255, 0),    # Yellow
            (204, 255, 102),  # Light Lime
            (153, 255, 51),   # Lime Green
            (102, 204, 0),    # Bright Green
            (0, 204, 102),    # Teal Green
            (0, 153, 153),    # Dark Teal
            (0, 102, 204),    # Cerulean Blue
            (51, 51, 102)     # Deep Navy
        ]


        rr.log(
            "world/objects",
            rr.Clear(recursive=True),
        )

        for i, single_obj in enumerate(object_list):
            obj_points, weights = single_obj.retrieve_valid_voxels_with_weights(diversity_percentile=self.vis_percentile_thresh, regularized=regularized)
            # obj_points = single_obj.retrieve_valid_voxels_clustered(diversity_percentile=self.vis_percentile_thresh)
            if len(obj_points) == 0:
                continue
            
            obj_label = single_obj.get_dominant_label()
            obj_name = f'{single_obj.obj_id}'.replace(" ", "")

            points = obj_points
            colors = np.array([colors_list[i]] * obj_points.shape[0])
            
            # # Weights as colors
            # weights_for_color = np.clip(weights, 1, 10) - 1
            # weights_for_color = weights_for_color.astype(int)
            # colors = np.array(point_colors)[weights_for_color]

            # # Cluster as colors
            # cluster_for_color = single_obj.clustering_labels
            # if cluster_for_color is not None:
            #     cluster_for_color = cluster_for_color[single_obj.valid_indices]
            #     assert len(cluster_for_color) == len(obj_points)
            #     cluster_for_color[cluster_for_color < 0] = 19
            #     cluster_for_color = np.clip(cluster_for_color, 0, 19)
            #     cluster_for_color = cluster_for_color.astype(int)
            #     colors = np.array(point_colors)[cluster_for_color]
            # else:
            #     cluster_for_color = np.zeros(obj_points.shape[0]).astype(int)
            #     colors = np.array(point_colors)[cluster_for_color]

            rr.log(
                f"world/objects/points/{obj_label}/{obj_name}/points",
                rr.Points3D(
                    points,
                    colors=colors,
                    radii=0.01,
                ),
            )

            if single_obj.obj_id[0] >= 0:
                bboxes_3d = single_obj.infer_bbox(diversity_percentile=self.vis_percentile_thresh, regularized=regularized)
                bboxes_oriented_3d = single_obj.infer_bbox_oriented(diversity_percentile=self.vis_percentile_thresh, regularized=regularized)

                if bboxes_3d is not None:
                    center, extent, q = bboxes_3d
                    half_extent = extent / 2
                    rr.log(
                        f"world/objects/bbox/{obj_label}/{obj_name}",
                        rr.Boxes3D(
                            centers=center,
                            half_sizes=half_extent,
                            colors=colors_list[i],
                            quaternions=rr.Quaternion(xyzw=q),
                        ),
                    )

                if bboxes_oriented_3d is not None:
                    center, extent, q = bboxes_oriented_3d
                    if extent is None:
                        continue
                    half_extent = extent / 2
                    rr.log(
                        f"world/objects/bbox_oriented/{obj_label}/{obj_name}",
                        rr.Boxes3D(
                            centers=center,
                            half_sizes=half_extent,
                            colors=colors_list[i],
                            quaternions=rr.Quaternion(xyzw=q),
                        ),
                    )

                centroid = single_obj.infer_centroid(diversity_percentile=self.vis_percentile_thresh, regularized=regularized)

                rr.log(
                    f"world/objects/centroid/{obj_label}/{obj_name}/",
                    rr.Points3D(
                        [centroid],
                        colors=[colors_list[i]],
                        radii=0.1,
                        labels=[f'{obj_label}_{obj_name}']
                    )
                )

                # centroid_2d = np.average(obj_points[:, :2], axis=0, weights=weights)
                # weighted_2d_points = obj_points[:, :2] - centroid_2d
                # weighted_2d_points = weighted_2d_points * np.sqrt(weights[:, np.newaxis])

                # pca = self.pca_solver.fit(weighted_2d_points)
                # direction_vector = pca.components_[0]
                # direction_vector = direction_vector / np.linalg.norm(direction_vector) * 1.0
                # direction_vector = np.concatenate([direction_vector, [0]])

                # rr.log(
                #     f"world/objects/{obj_label}_{single_obj.obj_id}/direction",
                #     rr.Arrows3D(
                #         origins=[np.mean(obj_points[:, :3], axis=0)],
                #         vectors=[direction_vector],
                #         colors=[colors_list[i]],
                #     )
                # )

    def visualize(self, object_list: list[SingleObject], odom, regularized=True, show_bbox=False):
        camera_to_world = rr.Transform3D(translation=odom['position'], rotation=rr.Quaternion(xyzw=odom['orientation']))
        rr.log("world/odom", camera_to_world)

        colors_list = generate_colors(len(object_list), is_int=False)
        point_colors = [
            (0, 0, 255),      # Deep Blue
            (51, 51, 255),    # Royal Blue
            (102, 102, 255),  # Light Blue
            (153, 153, 255),  # Periwinkle
            (204, 204, 255),  # Lavender
            (255, 204, 255),  # Light Purple
            (255, 153, 204),  # Pink
            (255, 102, 102),  # Light Red
            (255, 51, 51),    # Orange-Red
            (255, 0, 0),      # Bright Red
            (255, 128, 0),    # Deep Orange
            (255, 204, 0),    # Gold
            (255, 255, 0),    # Yellow
            (204, 255, 102),  # Light Lime
            (153, 255, 51),   # Lime Green
            (102, 204, 0),    # Bright Green
            (0, 204, 102),    # Teal Green
            (0, 153, 153),    # Dark Teal
            (0, 102, 204),    # Cerulean Blue
            (51, 51, 102)     # Deep Navy
        ]


        rr.log(
            "world/objects",
            rr.Clear(recursive=True),
        )

        for i, single_obj in enumerate(object_list):
            obj_points, weights = single_obj.retrieve_valid_voxels_with_weights(diversity_percentile=self.vis_percentile_thresh, regularized=regularized)
            # obj_points = single_obj.retrieve_valid_voxels_clustered(diversity_percentile=self.vis_percentile_thresh)
            if len(obj_points) == 0:
                continue
            
            obj_label = single_obj.get_dominant_label()
            obj_name = f'{single_obj.obj_id[0]}'

            points = obj_points
            colors = np.array([colors_list[i%len(colors_list)]] * obj_points.shape[0])
            
            # # Weights as colors
            # weights_for_color = np.clip(weights, 1, 10) - 1
            # weights_for_color = weights_for_color.astype(int)
            # colors = np.array(point_colors)[weights_for_color]

            # # Cluster as colors
            # cluster_for_color = single_obj.clustering_labels
            # if cluster_for_color is not None:
            #     cluster_for_color = cluster_for_color[single_obj.valid_indices]
            #     assert len(cluster_for_color) == len(obj_points)
            #     cluster_for_color[cluster_for_color < 0] = 19
            #     cluster_for_color = np.clip(cluster_for_color, 0, 19)
            #     cluster_for_color = cluster_for_color.astype(int)
            #     colors = np.array(point_colors)[cluster_for_color]
            # else:
            #     cluster_for_color = np.zeros(obj_points.shape[0]).astype(int)
            #     colors = np.array(point_colors)[cluster_for_color]

            rr.log(
                f"world/objects/points/{obj_label}/{obj_name}/points",
                rr.Points3D(
                    points,
                    colors=colors,
                    radii=0.01,
                ),
            )

            if show_bbox and single_obj.obj_id[0] >= 0:
                bboxes_3d = single_obj.infer_bbox(diversity_percentile=self.vis_percentile_thresh, regularized=regularized)
                bboxes_oriented_3d = single_obj.infer_bbox_oriented(diversity_percentile=self.vis_percentile_thresh, regularized=regularized)

                if bboxes_3d is not None:
                    center, extent, q = bboxes_3d
                    half_extent = extent / 2
                    rr.log(
                        f"world/objects/bbox/{obj_label}/{obj_name}",
                        rr.Boxes3D(
                            centers=center,
                            half_sizes=half_extent,
                            colors=colors_list[i],
                            quaternions=rr.Quaternion(xyzw=q),
                        ),
                    )

                if bboxes_oriented_3d is not None:
                    center, extent, q = bboxes_oriented_3d
                    if center is not None and extent is not None and q is not None:
                        half_extent = extent / 2
                        rr.log(
                            f"world/objects/bbox_oriented/{obj_label}/{obj_name}",
                            rr.Boxes3D(
                                centers=center,
                                half_sizes=half_extent,
                                colors=colors_list[i],
                                quaternions=rr.Quaternion(xyzw=q),
                            ),
                        )

                centroid = single_obj.infer_centroid(diversity_percentile=self.vis_percentile_thresh, regularized=regularized)

                rr.log(
                    f"world/objects/centroid/{obj_label}/{obj_name}/",
                    rr.Points3D(
                        [centroid],
                        colors=[colors_list[i]],
                        radii=0.1,
                        labels=[f'{obj_label}_{obj_name}']
                    )
                )

                # centroid_2d = np.average(obj_points[:, :2], axis=0, weights=weights)
                # weighted_2d_points = obj_points[:, :2] - centroid_2d
                # weighted_2d_points = weighted_2d_points * np.sqrt(weights[:, np.newaxis])

                # pca = self.pca_solver.fit(weighted_2d_points)
                # direction_vector = pca.components_[0]
                # direction_vector = direction_vector / np.linalg.norm(direction_vector) * 1.0
                # direction_vector = np.concatenate([direction_vector, [0]])

                # rr.log(
                #     f"world/objects/{obj_label}_{single_obj.obj_id}/direction",
                #     rr.Arrows3D(
                #         origins=[np.mean(obj_points[:, :3], axis=0)],
                #         vectors=[direction_vector],
                #         colors=[colors_list[i]],
                #     )
                # )
    
    def visualize_global_pcd(self, pcd_numpy):
        num_points = pcd_numpy.shape[0]
        colors = np.full((num_points, 4), [255, 255, 255, 48], dtype=np.uint8)
        rr.log(
            f"world/global_map",
            rr.Points3D(pcd_numpy,
                        colors=colors,
                        radii=0.01,
                        ),
        )

    def visualize_local_pcd_with_mesh(self, pcd_numpy):
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_numpy)
            pcd = pcd.voxel_down_sample(0.05)
            
            # obj_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=0.2)
            # vertices = np.asarray(obj_mesh.vertices)
            # # mesh_colors = np.array([colors_list[i]] * np.asarray(obj_mesh.vertices).shape[0])
            # # vertices_normals = np.asarray(obj_mesh.vertex_normals)
            # traingles = np.asarray(obj_mesh.triangles)

            rr.log(
                f"world/objects/local_map/points",
                rr.Points3D(
                    positions=pcd_numpy,
                    radii=0.02
                )
            )

            # rr.log(
            #     f"world/objects/local_map/mesh",
            #     rr.Mesh3D(
            #         vertex_positions=vertices,
            #         triangle_indices=traingles,
            #         # vertex_colors=mesh_colors
            #     )
            # )
        except Exception as e:
            print('vis local map failed')
            print(e)
            pass

    def save(self, filename):
        rr.save(filename)