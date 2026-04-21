#!/usr/bin/env python3
# coding: utf-8
import json
import rclpy
import base64
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
import torch
from torchvision import transforms
import cv2
import numpy as np
from std_msgs.msg import String, Int32
import google.generativeai as genai
from vlm_node.constants import GEMINI_API_KEY, MODEL_NAME, target_object, spatial_condition, attribute_condition, room_condition, anchor_object
from tare_planner.msg import RoomType, RoomEarlyStop1, VlmAnswer, ObjectType, NavigationQuery, TargetObjectInstruction, TargetObject, TargetObjectWithSpatial
from rviz_2d_overlay_msgs.msg import OverlayText
from openai import OpenAI
from pydantic import BaseModel
import os
import json
import time
from collections import deque
import threading
import yaml
from rclpy.time import Time
from vlm_node.utils import project_bbox3d
import yaml


class VLMNode(Node):
    def __init__(self):
        super().__init__('vlm_node')

        # Initialize VLM
        self.declare_parameter('log_dir', 'logs/episode_0')
        self.declare_parameter('platform', 'mecanum')

        self.log_dir = self.get_parameter('log_dir').get_parameter_value().string_value
        self.platform = self.get_parameter('platform').get_parameter_value().string_value

        self.get_logger().info(f"Log directory: {self.log_dir}")

        try:
            genai.configure(api_key=GEMINI_API_KEY)
            self.vlm_model = OpenAI(
                api_key=GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            self.get_logger().info("✅ VLM initialized successfully")
        except Exception as e:
            self.get_logger().error(f"❌ VLM initialization failed: {e}")
            return

        # self.room_type_vlm_model = "gemini-2.5-flash-lite"  # Use the flash lite model for faster response
        self.room_type_vlm_model = "gemini-2.5-flash" 
        self.room_nav_vlm_model = "gemini-2.5-flash"  
        # TODO: Unit Test
        self.object_type_vlm_model = "gemini-2.5-flash-lite"  
        
        # queues
        self.room_type_query_queue = deque()
        self.room_navigation_query_queue = deque(maxlen=1)  # only keep the latest query
        self.room_early_stop_1_query_queue = deque()
        self.object_type_query_queue = deque()
        self.instruction_queue = deque()
        self.target_object_query_queue = deque()
        self.target_object_spatial_query_queue = deque()
        self.anchor_object_query_queue = deque()
        self.target_object_counter = 0
        self.target_object_spatial_counter = 0
        self.anchor_object_counter = 0

        # Simulation room types
        # self.room_types = ["Living Room", "Bedroom", "Kitchen", "Bathroom", "Balcony", "Garden"]
        # Gates 4th floor room_types
        # self.room_types = ["Classroom", "Office Room", "Computer Lab", "Restroom", "Student Lounge", "Reception", "Corridor"]
        # Gates 5th floor room_types
        # self.room_types = ["Classroom", "Office Room", "Meeting Room", "Computer Lab", "Restroom", "Storage Room", "Copy Room", "Student Lounge", "Reception", "Corridor"]
        # self.room_types = ["Classroom", "Computer Lab", "Restroom", "Student Lounge", "Corridor"]
        # NSH room_types
        self.room_types = ["Classroom", "Laboratory", "Office Room", "Meeting Room", "Computer Lab", "Restroom", "Storage Room", "Copy Room", "Student Lounge", "Reception", "Corridor"]
        # self.room_types = ["Office Room"]
        # CIC room_types
        # self.room_types = ["Office Room", "Meeting Room", "Open Workspace", "Interview Room", "Reception", "Print Room", "Storage Room", "Restroom"]
        self.ROOM_TYPE_PROMPT = """
        You are given an rgb image of a room and a top-down room layout mask image.
        Identify the room type and respond strictly in valid JSON format.

        Use the key "room_type" and select one of the options listed below as the value.  
        For example:
        {"room_type": "Living Room"}

        Options:
        """

        self.ROOM_FINISH_NAVIGATION_PROMPT = """
        You are a wheeled mobile robot operating in an indoor environment. Your goal is to efficiently find a target object based on a human-provided instruction in a new house. The current room you are in has been fully explored. To achieve the goal, you must select the next room to explore from rooms listed in a JSON file, aiming to complete the task as quickly as possible.

        ### Provided Information:
        1. A specific instruction describing the task.
        2. A JSON file containing details of each candidate room.
        3. An image of each candidate room.

        ### JSON File Structure:
        - **Rooms**
        - 'room_id': A unique identifier for the room.
        - 'label': The type of room (e.g., "Living Room", "Bedroom", etc.).
        - 'objects': A list of objects detected in the room.
        - 'distance': The distance (in meters) the robot needs to travel to reach this room.

        ### Task:
        You must carefully analyze the JSON file, using logical reasoning and common sense, to select the next room to explore from the list of rooms. Consider the following factors:
        - Evaluate how closely each room aligns with the overall task objective.
        - Optimize the exploration path by minimizing unnecessary backtracking or redundant movements.
        - Assess the likelihood that exploring the selected room will meaningfully advance or complete the overall task.

        ### Output Format:
        Your response should include:
        - 'steps': The chain of thought leading to the decision.
        - `final_answer`: The `room id` of the next room to explore.
        - `reason`: The rationale for selecting this room.
        """

        self.room_early_stop_1_prompt = """
        You are a wheeled mobile robot operating in an indoor environment. Your task is to efficiently locate a target object. You will be provided with an image of a nearby room and an image of the current room you are exploring. Based on these observations, you must decide whether to stop exploring the current room and move to the other room in order to find the target object as quickly as possible. If you believe that the nearby room is more likely to contain the target object and decide to early stop, you should return `early_stop` as `True`.
        
        Besides, you should first check whether the image of the nearby room contains the target object(ignore all the spatial, self-attribute and room constrain, only considering the object type). If the nearby room image contains the target object, you should also return `early_stop` as `True`.
        """

        self.object_type_query_prompt = """
        You are provided with an RGB image containing an object within a bounding box, along with a list of candidate labels. Your task is to determine the correct label for the object based on the image and select the most appropriate option from the list. The response must be strictly in valid JSON format.
        Use the key "label" and select one of the provided labels as the value.  
        For example:
        {"label": "Chair"}
        """

        self.instruction_decomposition_prompt = """
        You are given a human-provided instruction describing a task to find a target object in an indoor environment. Your task is to decompose the instruction into five components: the target object, the room condition, the spatial condition, the attribute condition of the target object, the anchor object and the attribute condition of the anchor object. If room, spatial, attribute conditions or anchor object are not specified, return them as empty strings. The response must be strictly in valid JSON format.
        Use the keys "target_object", "room_condition", "spatial_condition", "attribute_condition", "anchor object" and "attribute_condition_anchor" for the respective components.
        For example:
        {"Instruction": "Find a silver refrigerator next to the white cabinet in the student lounge."}
        {"target_object": "refrigerator", "room_condition": "in the student lounge", "spatial_condition": "next to the white cabinet", "attribute_condition": "silver", "anchor_object": "cabinet", "attribute_condition_anchor": "white"}
        """

        self.target_object_prompt = """
        You are given an instruction, specifying object type, room condition and self-attribute condition, to locate a target object in an indoor environment and the description, describing the object type and room location, of a candidate object, along with a close-up image of the object. If the room location is None, please infer the room position of the object from the provided image. If the room location is given, you can directly use it. Please infer the room position and the self-attribute of the object from the provided image.

        Your task is to decide whether the detected object matches the target object specified in the instruction. Consider both attribute conditions and room conditions.

        If any condition is not satisfied or cannot be clearly determined, return "is_target": false.

        If all conditions are satisfied, return "is_target": true.

        The output must be strictly in valid JSON format.
        Example:
        {"is_target": true, "reason": "The object is a silver refrigerator located in the student lounge, which matches the instruction."}
        """

        self.target_object_spatial_prompt = """
        You are given an instruction, specifying object type, room condition, spatial condition and self-attribute condition, to locate a target object in an indoor environment and the description, describing the object type and room location, of a candidate object, along with a close-up image of the object, and several images of the object and its surroundings captured from different viewpoints. If the room location is None, please infer the room position of the object from the provided image. If the room location is given, you can directly use it. Please infer the spatial position and the self-attribute of the object from the provided image.

        Your task is to decide whether the detected object matches the target object specified in the instruction. Consider both attribute conditions, spatial conditions and room conditions.

        If any condition is not satisfied or cannot be clearly determined, return "is_target": false.

        If all conditions are satisfied, return "is_target": true.

        The output must be strictly in valid JSON format.
        Example:
        {"is_target": true, "reason": "The object is a silver refrigerator located in the student lounge, which matches the instruction."}
        """

        object_file_path = 'src/semantic_mapping/semantic_mapping/config/objects.yaml'
        with open(object_file_path, "r") as file:
            self.object_config = yaml.safe_load(file)
        self.label_template = self.object_config['prompts']
        self.object_list = []
        for value in self.label_template.values():
            self.object_list += value['prompts']
        self.get_logger().info(f"Object List: {self.object_list}")

        self.skip_set = ["next", "dynamic"]

        self.bridge = CvBridge()

        # ----------------------- Subscribers -----------------------
        # Subscriber: receive the room type query
        self.room_type_query_subscription = self.create_subscription(
            RoomType,
            '/room_type_query',
            self.room_type_callback,
            10
        )

        self.room_navigation_subscription = self.create_subscription(
            NavigationQuery,
            '/room_navigation_query',
            self.room_navigation_callback,
            5
        )

        self.room_early_stop_1_subscription = self.create_subscription(
            RoomEarlyStop1,
            '/room_early_stop_1',
            self.room_early_stop_1_callback,
            5
        )

        self.object_type_query_subscription = self.create_subscription(
            ObjectType,
            '/object_type_query',
            self.object_type_query_callback,
            50
        )

        self.instruction_subscription = self.create_subscription(
            String,
            '/keyboard_input',
            self.instruction_callback,
            10
        )

        self.target_object_subscription = self.create_subscription(
            TargetObject,
            '/target_object_query',
            self.target_object_query_callback,
            10
        )

        self.target_object_spatial_subscription = self.create_subscription(
            TargetObjectWithSpatial,
            '/target_object_spatial_query',
            self.target_object_spatial_query_callback,
            10
        )

        self.anchor_object_subscription = self.create_subscription(
            TargetObject,
            '/anchor_object_query',
            self.anchor_object_query_callback,
            10
        )
        
        # ----------------------- End of Subscribers ---------------------

        # ----------------------- Publishers -----------------------
        # Publisher: publish room type answer
        self.room_type_publisher = self.create_publisher(
            RoomType,
            '/room_type_answer',
            10
        )
        # Publisher: publish room navigation answer
        self.room_navigation_publisher = self.create_publisher(
            VlmAnswer,
            '/room_navigation_answer',
            5
        )
        # # Publisher: publish room early stop 1 answer
        # self.room_early_stop_1_publisher = self.create_publisher(
        #     RoomEarlyStop1,
        #     '/room_early_stop_1_answer',
        #     5
        # )
        self.text_overlay_publisher = self.create_publisher(
            OverlayText,
            '/vlm_answer',
            10
        )

        self.object_type_answer_publisher = self.create_publisher(
            ObjectType,
            '/object_type_answer',
            50
        )
        
        self.target_object_instruction_publisher = self.create_publisher(
            TargetObjectInstruction,
            '/target_object_instruction',
            10
        )

        self.target_object_answer_publisher = self.create_publisher(
            TargetObject,
            '/target_object_answer',
            10
        )
        self.anchor_object_answer_publisher = self.create_publisher(
            TargetObject,
            '/anchor_object_answer',
            10
        )
        # ----------------------- End of Publishers ---------------------

        self._last_msg_id = 0   # 最新消息的ID

        self.mapping_timer = self.create_timer(0.1, self.vlm_node_callback)

        # debug
        os.system("rm -rf debug")
        os.makedirs("debug", exist_ok=True)
        os.makedirs("debug/room_type", exist_ok=True)
        os.makedirs("debug/room_navigation", exist_ok=True)
        os.makedirs("debug/room_early_stop_1", exist_ok=True)
        os.makedirs("debug/object_type", exist_ok=True)
        os.makedirs(f"debug/target_object", exist_ok=True)
        os.makedirs(f"debug/anchor_object", exist_ok=True)
        os.makedirs(f"debug/target_object_spatial", exist_ok=True)

        self.viewpoint_path = "output/viewpoint_images"
        
        # Target object from constants
        self.target_object = target_object
        self.room_condition = room_condition
        self.spatial_condition = spatial_condition
        self.anchor_object = anchor_object
        self.attribute_condition = attribute_condition
        self.instruction = f"Find a {self.attribute_condition} {self.target_object} {self.spatial_condition} {self.room_condition}."

        # Anchor object
        self.anchor_object = ""
        self.attribute_condition_anchor = ""
        self.instruction_anchor = f"Find a {self.attribute_condition_anchor} {self.anchor_object} {self.room_condition}."

        # publish the initial target object
        target_object_instruction_msg = TargetObjectInstruction()
        target_object_instruction_msg.target_object = self.target_object
        target_object_instruction_msg.room_condition = self.room_condition
        target_object_instruction_msg.spatial_condition = self.spatial_condition
        target_object_instruction_msg.attribute_condition = self.attribute_condition
        target_object_instruction_msg.anchor_object = self.anchor_object
        target_object_instruction_msg.attribute_condition_anchor = self.attribute_condition_anchor
        self.target_object_instruction_publisher.publish(target_object_instruction_msg)

        self.get_logger().info("🚀 VLM Node started")
        self.get_logger().info(f"🎯 Target object: {self.target_object}")
        
    
    def room_type_callback(self, msg: RoomType):
        self.room_type_query_queue.append(msg)

    def room_navigation_callback(self, msg: NavigationQuery):
        self.room_navigation_query_queue.append(msg)

    def room_early_stop_1_callback(self, msg: RoomEarlyStop1):
        self.room_early_stop_1_query_queue.append(msg)
    
    def object_type_query_callback(self, msg: ObjectType):
        self.object_type_query_queue.append(msg)
    
    def instruction_callback(self, msg: String):
        self.instruction_queue.append(msg.data)
    
    def target_object_query_callback(self, msg: TargetObject):
        self.target_object_query_queue.append(msg)

    def target_object_spatial_query_callback(self, msg: TargetObjectWithSpatial):
        self.target_object_spatial_query_queue.append(msg)
    
    def anchor_object_query_callback(self, msg: TargetObject):
        self.anchor_object_query_queue.append(msg)

    def publish_text_overlay(self, text: str, duration: float = 8.0):
        self._last_msg_id += 1
        msg_id = self._last_msg_id

        overlay_msg = OverlayText()
        overlay_msg.text = text
        overlay_msg.width = 640
        overlay_msg.height = 100
        overlay_msg.text_size = 24
        self.text_overlay_publisher.publish(overlay_msg)

        def clear_text():
            # 只有当自己还是最新消息时才清空
            if msg_id == self._last_msg_id:
                clear_msg = OverlayText()
                clear_msg.text = ""
                clear_msg.width = 640
                clear_msg.height = 100
                clear_msg.text_size = 24
                self.text_overlay_publisher.publish(clear_msg)

        threading.Timer(duration, clear_text).start()

    def process_room_type_query(self, msg: RoomType):
        """Handle room type query and publish answer"""
        class Step(BaseModel):
            explanation: str
            output: str

        class Result(BaseModel):
            # steps: list[Step]
            room_type: str
        try:
            # Process the room type query
            # self.get_logger().info(f"Received room type query: {msg}")
            start_time = time.time()
            # transform the sensor_msgs::msg::Image to a format suitable for VLM
            image_data = msg.image  # Assuming msg.image contains the image data
            if not image_data:
                raise ValueError("No image data provided in the room type query")
            cv_image = self.bridge.imgmsg_to_cv2(image_data, desired_encoding='bgr8')
            cv_room_mask = self.bridge.imgmsg_to_cv2(msg.room_mask, desired_encoding='mono8')
            img_jpg = cv2.imencode('.jpg', cv_image)[1]
            room_mask = cv2.imencode('.jpg', cv_room_mask)[1]
            img_base64 = base64.b64encode(img_jpg).decode('utf-8')
            room_mask_base64 = base64.b64encode(room_mask).decode('utf-8')
            if msg.in_room:
                room_type_prompt = self.ROOM_TYPE_PROMPT
                room_types = self.room_types.copy()
                for i, room_type in enumerate(room_types):
                    room_type_prompt += f"{i}. {room_type}\n"
            else:
                room_type_prompt = self.ROOM_TYPE_PROMPT
                room_types = self.room_types.copy()
                if "Corridor" in room_types:
                    room_types.remove("Corridor")
                for i, room_type in enumerate(room_types):
                    room_type_prompt += f"{i}. {room_type}\n"

            completion = self.vlm_model.beta.chat.completions.parse(
                model=self.room_type_vlm_model, # Use the flash lite model for faster response
                messages=[{
                    "role": "system",
                    "content": room_type_prompt
                }, {
                    "role":
                        "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{room_mask_base64}"}},
                        # {"type": "text", "text": prompt_info}
                    ]
                }],
                response_format=Result,
            )
            # transform the image data to a format suitable for VLM
            try:
                answer = completion.choices[0].message.parsed
            except Exception:
                raw_text = completion.choices[0].message.content.strip()
                # 如果只是普通字符串，就包一层 JSON
                answer = Result(room_type=raw_text)
            # print the answer
            self.get_logger().info(f"Received room type answer: {answer}")
            room_type = answer.room_type
            self.get_logger().info(f"Determined room type: {room_type}")
            # Publish the room type answer
            answer_msg = msg
            answer_msg.room_type = room_type.lower()
            self.room_type_publisher.publish(answer_msg)
            # self.get_logger().info("Published room type answer")
            end_time = time.time()
            self.get_logger().info(f"Room type query processed in {end_time - start_time:.2f} seconds")

            # save the image and room mask for debugging
            img_path = f"debug/room_type/{msg.room_id}_{room_type}.jpg"
            mask_path = f"debug/room_type/{msg.room_id}_{room_type}_mask.jpg"
            cv2.imwrite(img_path, cv_image)
            cv2.imwrite(mask_path, cv_room_mask)
            # save the answer to a text file for debugging
            answer_file_path = f"debug/room_type/{msg.room_id}_{room_type}.txt"
            with open(answer_file_path, 'w') as f:
                f.write(f"Room ID: {msg.room_id}\nIn Room: {msg.in_room}\nAnswer: {answer}\n")
        
        except Exception as e:
            self.get_logger().error(f"Error processing room type query: {e}")
    
    def process_room_navigation_query(self, msg: NavigationQuery):
        """Handle room navigation query(receive a JSON string) and publish answer"""
        dumper_string = msg.json
        json_file = json.loads(dumper_string)
        candidate_rooms = []
        candidate_rooms_json = []
        candidate_rooms_image = []
        candidate_rooms_image_jpg = []
        candidate_rooms_anchor_point = {}
        for idx, room in enumerate(json_file['rooms']):
            candidate_rooms.append(room['room id'])
            candidate_rooms_json.append(room)
            candidate_rooms_anchor_point[room['room id']] = msg.anchor_points[idx]

        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)  # 黑色图像
        blank_image_jpg = cv2.imencode('.jpg', blank_image)[1]
        blank_image_base64 = base64.b64encode(blank_image_jpg).decode('utf-8')
        for img_msg in msg.images:
            if not img_msg.data:
                # push a blank cv image
                candidate_rooms_image.append(blank_image_base64)
                candidate_rooms_image_jpg.append(blank_image_jpg)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
                img_jpg = cv2.imencode('.jpg', cv_image)[1]
                img_base64 = base64.b64encode(img_jpg).decode('utf-8')
                candidate_rooms_image.append(img_base64)
                candidate_rooms_image_jpg.append(img_jpg)

        self.get_logger().info(f"Received room navigation query with candidate rooms: {candidate_rooms}")

        if not candidate_rooms:
            self.get_logger().error("No candidate rooms provided in the room navigation query")
            return
        
        # # test: randomly select a room from the candidate rooms
        # import random
        # selected_room = random.choice(candidate_rooms)
        # # self.get_logger().info(f"Selected room for navigation: {selected_room}")
        # # Publish the selected room as an answer
        # answer_msg = Int32()
        # answer_msg.data = selected_room
        # self.room_navigation_publisher.publish(answer_msg)
        # self.get_logger().info(f"Published room navigation answer: {selected_room}")

        class Step(BaseModel):
            explanation: str
            output: str
        class Result(BaseModel):
            steps: list[Step]
            final_answer: int
            reason: str
        try:
            content_blocks = [
                {"type": "text", "text": f"Instruction: {self.instruction}"},
                {"type": "text", "text": "Here are the candidate rooms with descriptions and photos."},
            ]

            for room, img_base64 in zip(candidate_rooms_json, candidate_rooms_image):
                # 先添加文字描述
                room_desc = json.dumps(room, ensure_ascii=False)  # 或者只挑选关键信息
                content_blocks.append({"type": "text", "text": room_desc})

                # 再添加对应的图像
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                })
            # Process the room navigation query
            completion = self.vlm_model.beta.chat.completions.parse(
                model=self.room_nav_vlm_model,
                messages=[{
                    "role": "system",
                    "content": self.ROOM_FINISH_NAVIGATION_PROMPT
                }, {
                    "role": "user",
                    "content": content_blocks
                }],
                response_format=Result,
            )
            answer = completion.choices[0].message.parsed
            # print the answer
            self.get_logger().info(f"Received room navigation answer: {answer}")
            selected_room = answer.final_answer
            self.get_logger().info(f"Selected room for navigation: {selected_room}")
            # Publish the selected room as an answer
            answer_msg = VlmAnswer()
            answer_msg.room_id = selected_room
            answer_msg.anchor_point = candidate_rooms_anchor_point[selected_room]
            answer_msg.answer_type = 0  # 0 for room navigation
            self.room_navigation_publisher.publish(answer_msg)
            self.get_logger().info(f"Published room navigation answer: {selected_room}")

            # publish the reason to the text overlay
            text = f"Next Room ID: {selected_room}" + f"\nReason: {answer.reason}" + "\nThinking Steps:\n" + '\n'.join([f"{step.explanation}" for step in answer.steps])
            self.publish_text_overlay(text)

            # save the input JSON and answer to 2 text files for debugging, using timestamp to avoid overwriting
            time_int = int(time.time())
            os.makedirs(f"debug/room_navigation/{time_int}", exist_ok=True)
            json_file_path = f"debug/room_navigation/{time_int}/room_navigation_query.json"
            answer_file_path = f"debug/room_navigation/{time_int}/room_navigation_answer.txt"
            with open(json_file_path, 'w') as f:
                f.write(dumper_string)
            with open(answer_file_path, 'w') as f:
                f.write(f"Answer: {answer}\n")
            # save the candidate room images
            for i, img_jpg in enumerate(candidate_rooms_image_jpg):
                img_path = f"debug/room_navigation/{time_int}/room_{candidate_rooms[i]}.jpg"
                cv2.imwrite(img_path, cv2.imdecode(img_jpg, cv2.IMREAD_COLOR))

        except Exception as e:
            self.get_logger().error(f"Error processing room navigation query: {e}")
    
    def process_room_early_stop_1_query(self, msg: RoomEarlyStop1):
        """Handle room early stop 1 query and publish answer"""
        class Step(BaseModel):
            explanation: str
            output: str

        class Result(BaseModel):
            steps: list[Step]
            early_stop: bool

        # # save image for debugging
        # img1 = self.bridge.imgmsg_to_cv2(msg.image_1, desired_encoding='bgr8')
        # img2 = self.bridge.imgmsg_to_cv2(msg.image_2, desired_encoding='bgr8')
        # img1_path = f"debug/room_early_stop_1_img_1_{msg.anchor_point.x}_{msg.anchor_point.y}.jpg"
        # img2_path = f"debug/room_early_stop_1_img_2_{msg.anchor_point.x}_{msg.anchor_point.y}.jpg"
        # cv2.imwrite(img1_path, img1)
        # cv2.imwrite(img2_path, img2)
        try:
            # Process the room early stop 1 query
            # self.get_logger().info(f"Received room early stop 1 query: {msg}")
            start_time = time.time()
            room_id_1 = msg.room_id_1
            room_id_2 = msg.room_id_2

            img1 = self.bridge.imgmsg_to_cv2(msg.image_1, desired_encoding='bgr8')
            img2 = self.bridge.imgmsg_to_cv2(msg.image_2, desired_encoding='bgr8')
            img1_jpg = cv2.imencode('.jpg', img1)[1]
            img2_jpg = cv2.imencode('.jpg', img2)[1]
            img1_base64 = base64.b64encode(img1_jpg).decode('utf-8')
            img2_base64 = base64.b64encode(img2_jpg).decode('utf-8')

            instruction = f"Instruction: {self.instruction}\n The first image is a nearby room, and the second image is the current room you are exploring."
            completion = self.vlm_model.beta.chat.completions.parse(
                model=self.room_nav_vlm_model, # Use the flash lite model for faster response
                messages=[{
                    "role": "system",
                    "content": self.room_early_stop_1_prompt
                }, {
                    "role":
                        "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_base64}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_base64}"}},
                    ]
                }],
                response_format=Result,
            )
            answer = completion.choices[0].message.parsed
            # print the answer
            self.get_logger().info(f"Received room early stop 1 answer: {answer}")
            early_stop = answer.early_stop
            # if early_stop:
            # TODO: Unit Test
            if False:
            # if True:
                self.get_logger().info("Early stop is True, stopping exploration")
                # Publish the room early stop 1 answer
                answer_msg = VlmAnswer()
                answer_msg.room_id = room_id_1  # move to room_id_1
                answer_msg.anchor_point = msg.anchor_point_1
                answer_msg.answer_type = 1  # 1 for early stop 1
                self.room_navigation_publisher.publish(answer_msg)
            else:
                self.get_logger().info("Early stop is False, continuing exploration")
                if msg.enter_wrong_room:
                    answer_msg = VlmAnswer()
                    answer_msg.room_id = room_id_2  # continue to explore room_id_2
                    answer_msg.anchor_point = msg.anchor_point_2
                    answer_msg.answer_type = 1  # 1 for early stop 1
                    self.room_navigation_publisher.publish(answer_msg)
            # # Publish the room early stop 1 answer
            # answer_msg = msg
            # answer_msg.early_stop_1 = early_stop
            # self.room_early_stop_1_publisher.publish(answer_msg)
            end_time = time.time()
            self.get_logger().info(f"Room early stop 1 query processed in {end_time - start_time:.2f} seconds")
            
            # publish the reason to the text overlay
            text = f"Early Stop Decision: {early_stop}\nReason: {'; '.join([step.explanation for step in answer.steps])}"
            self.publish_text_overlay(text)

            # save the input images and answer to 2 text files for debugging, using timestamp to avoid overwriting
            time_int = int(time.time())
            os.makedirs(f"debug/room_early_stop_1/{room_id_1}_{time_int}", exist_ok=True)
            img1_path = f"debug/room_early_stop_1/{room_id_1}_{time_int}/room_early_stop_1_img_1.jpg"
            img2_path = f"debug/room_early_stop_1/{room_id_1}_{time_int}/room_early_stop_1_img_2.jpg"
            answer_file_path = f"debug/room_early_stop_1/{room_id_1}_{time_int}/room_early_stop_1_answer.txt"
            cv2.imwrite(img1_path, img1)
            cv2.imwrite(img2_path, img2)
            with open(answer_file_path, 'w') as f:
                f.write(f"Answer: {answer}\n")
        
        except Exception as e:
            self.get_logger().error(f"Error processing room early stop 1 query: {e}")
            return
    
    def process_object_type_query(self, msg: ObjectType):
        """Handle target object query and publish answer"""
        img = np.load(msg.img_path)
        mask_path = msg.img_path.replace('.npy', '_mask.npy')
        if os.path.exists(mask_path):
            mask = np.load(mask_path, mmap_mode=None)
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
            # img = cv2.bitwise_and(img, img, mask=mask)

            # find contours of the mask and draw them
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # draw the contours on the image for debugging
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        img_jpg = cv2.imencode('.jpg', img)[1]
        img_base64 = base64.b64encode(img_jpg).decode('utf-8')
        labels = msg.labels
        class Result(BaseModel):
            reason: str
            label: str
        try:
            # Process the target object query
            completion = self.vlm_model.beta.chat.completions.parse(
                model=self.object_type_vlm_model,
                messages=[{
                    "role": "system",
                    "content": self.object_type_query_prompt
                }, {
                    "role":
                        "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                        {"type": "text", "text": f"Possible labels: {', '.join(self.object_list)}"},
                    ]
                }],
                response_format=Result,
            )
            answer = completion.choices[0].message.parsed
            # print the answer
            self.get_logger().info(f"Received target object answer: {answer}")
            verified_label = answer.label
            self.get_logger().info(f"Verified target object label: {verified_label}")
            # Publish the target object answer
            answer_msg = ObjectType()
            answer_msg.object_id = msg.object_id
            answer_msg.img_path = msg.img_path
            answer_msg.final_label = verified_label.lower()
            answer_msg.labels = msg.labels
            self.object_type_answer_publisher.publish(answer_msg)
            self.get_logger().info("Published target object answer")

            text = f"Object ID: {msg.object_id}\nVerified Label: {verified_label}\nPossible Labels: {', '.join(labels)}"
            # self.publish_text_overlay(text)

            # save the image for debugging
            img_path = f"debug/object_type/{msg.object_id}_{verified_label}.jpg"
            cv2.imwrite(img_path, img)
            # save the answer to a text file for debugging
            answer_file_path = f"debug/object_type/{msg.object_id}_{verified_label}.txt"
            with open(answer_file_path, 'w') as f:
                f.write(f"Verified Label: {verified_label}\nReason: {answer.reason}")
                
        except Exception as e:
            self.get_logger().error(f"Error processing target object query: {e}")
    
    def process_instruction(self, instruction: str):
        """Process new instruction from keyboard input and decompose it into target object, spatial condition, and attribute condition"""
        if instruction in self.skip_set:
            self.get_logger().info(f"Skipping instruction: {instruction}")
            return
        
        class Result(BaseModel):
            target_object: str
            room_condition: str
            spatial_condition: str
            attribute_condition: str
            anchor_object: str
            attribute_condition_anchor: str
        try:
            completion = self.vlm_model.beta.chat.completions.parse(
                model=self.object_type_vlm_model,
                messages=[{
                    "role": "system",
                    "content": self.instruction_decomposition_prompt
                }, {
                    "role":
                        "user",
                    "content": [
                        {"type": "text", "text": f"Instruction: {instruction}"},
                    ]
                }],
                response_format=Result,
            )
            answer = completion.choices[0].message.parsed
            # print the answer
            self.get_logger().info(f"Decomposed instruction: {answer}")
            self.target_object = answer.target_object.lower()
            self.room_condition = answer.room_condition.lower()
            self.spatial_condition = answer.spatial_condition.lower()
            self.attribute_condition = answer.attribute_condition.lower()
            self.anchor_object = answer.anchor_object.lower()
            self.attribute_condition_anchor = answer.attribute_condition_anchor.lower()
            self.instruction = instruction.lower()
            self.instruction_anchor = f"Find a {self.attribute_condition_anchor} {self.anchor_object} {self.room_condition}."

            # publish the new target object to /target_object topic
            target_object_instruction_msg = TargetObjectInstruction()
            target_object_instruction_msg.target_object = self.target_object
            target_object_instruction_msg.room_condition = self.room_condition
            target_object_instruction_msg.spatial_condition = self.spatial_condition
            target_object_instruction_msg.attribute_condition = self.attribute_condition
            target_object_instruction_msg.anchor_object = self.anchor_object
            target_object_instruction_msg.attribute_condition_anchor = self.attribute_condition_anchor
            self.target_object_instruction_publisher.publish(target_object_instruction_msg)
            self.get_logger().info("Published new target object")
            text = f"New Instruction: {instruction}\nTarget Object: {self.target_object}\nSpatial Condition: {self.spatial_condition}\nAttribute Condition: {self.attribute_condition}\nAnchor Object: {self.anchor_object}\nAttribute Condition Anchor: {self.attribute_condition_anchor}"
            self.publish_text_overlay(text)

            # save the instruction and decomposition result to a text file for debugging
            os.makedirs("debug/instruction_decomposition", exist_ok=True)
            answer_file_path = f"debug/instruction_decomposition/{int(time.time())}.txt"
            with open(answer_file_path, 'w') as f:
                f.write(f"Instruction: {instruction}\nDecomposed Result: {answer}\n")

        except Exception as e:
            self.get_logger().error(f"Error processing instruction: {e}")
    
    def process_target_object_query(self, msg: TargetObject):
        """Handle target object query and publish answer"""
        img = np.load(msg.img_path)
        mask_path = msg.img_path.replace('.npy', '_mask.npy')
        if os.path.exists(mask_path):
            mask = np.load(mask_path, mmap_mode=None)
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
            # img = cv2.bitwise_and(img, img, mask=mask)

            # find contours of the mask and draw them
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # draw the contours on the image for debugging
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        img_jpg = cv2.imencode('.jpg', img)[1]
        img_base64 = base64.b64encode(img_jpg).decode('utf-8')

        object_label = msg.object_label
        room_label = msg.room_label
        if room_label == "":
            description = f"This is a {object_label}. {{ 'Obejct Type': {object_label}, 'Room Type': Unknown }}, Here is its image."
        else:        
            description = f"This is a {object_label} in the {room_label}. {{ 'Obejct Type': {object_label}, 'Room Type': {room_label} }}, Here is its image."

        instruction = f"Instruction: {self.instruction}, {{ 'Target Object': {self.target_object}, 'Room Condition': {self.room_condition if self.room_condition else 'None'}, 'Attribute Condition': {self.attribute_condition if self.attribute_condition else 'None'} }} \n Determine whether the following object matches the target object described in the instruction."

        class Result(BaseModel):
            reason: str
            is_target: bool
        try:
            # Process the target object query
            completion = self.vlm_model.beta.chat.completions.parse(
                model=self.object_type_vlm_model,
                messages=[{
                    "role": "system",
                    "content": self.target_object_prompt
                }, {
                    "role":
                        "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "text", "text": description},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    ]
                }],
                response_format=Result,
            )
            answer = completion.choices[0].message.parsed
            # print the answer
            self.get_logger().info(f"Received target object answer: {answer}")
            is_target = answer.is_target
            self.get_logger().info(f"Is target object: {is_target}")
            # Publish the target object answer
            answer_msg = TargetObject()
            answer_msg.header = msg.header
            answer_msg.object_id = msg.object_id
            answer_msg.img_path = msg.img_path
            answer_msg.object_label = msg.object_label
            answer_msg.room_label = msg.room_label
            answer_msg.is_target = is_target
            self.target_object_answer_publisher.publish(answer_msg)
            self.get_logger().info("Published target object answer")

            text = f"Object ID: {msg.object_id}\nObject Label: {object_label}\nRoom Label: {room_label}\nIs Target: {is_target}\nReason: {answer.reason}"
            self.publish_text_overlay(text)
            # save the image for debugging
            time_int = int(time.time())
            img_path = f"debug/target_object/{time_int}_{msg.object_id}_{object_label}_{is_target}.jpg"
            cv2.imwrite(img_path, img)
            # save the answer to a text file for debugging
            answer_file_path = f"debug/target_object/{time_int}_{msg.object_id}_{object_label}_{is_target}.txt"
            with open(answer_file_path, 'w') as f:
                f.write(f"Instruction: {instruction}\nDescription: {description}\n")
                f.write(f"Is Target: {is_target}\nReason: {answer.reason}")
        except Exception as e:
            self.get_logger().error(f"Error processing target object query: {e}")
    
    def process_target_object_spatial_query(self, msg: TargetObjectWithSpatial):
        """Handle target object with spatial query and publish answer"""
        # debug
        viewpoint_ids = msg.viewpoint_ids
        viewpoint_imgs = []
        viewpoint_img_bases = []
        for vp_id in viewpoint_ids:
            if vp_id <= 0:
                continue
            vp_image_path = os.path.join(self.viewpoint_path, f"viewpoint_{vp_id}.png")
            vp_pose_path = os.path.join(self.viewpoint_path, f"viewpoint_{vp_id}_transform.npy")
            if not os.path.exists(vp_image_path):
                self.get_logger().warning(f"Viewpoint image {vp_image_path} does not exist")
                continue
            viewpoint_image = cv2.imread(vp_image_path)
            viewpoint_transform_matrix = np.load(vp_pose_path)
            bbox3d = []
            for point in msg.bbox3d:
                bbox3d.append([point.x, point.y, point.z])
            projected_img = project_bbox3d(viewpoint_image, bbox3d, viewpoint_transform_matrix, self.platform)
            projected_img_jpg = cv2.imencode('.jpg', projected_img)[1]
            projected_img_base64 = base64.b64encode(projected_img_jpg).decode('utf-8')

            viewpoint_imgs.append(projected_img)
            viewpoint_img_bases.append(projected_img_base64)
        
        img = np.load(msg.img_path)
        mask_path = msg.img_path.replace('.npy', '_mask.npy')
        if os.path.exists(mask_path):
            mask = np.load(mask_path, mmap_mode=None)
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
            # img = cv2.bitwise_and(img, img, mask=mask)

            # find contours of the mask and draw them
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # draw the contours on the image for debugging
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        img_jpg = cv2.imencode('.jpg', img)[1]
        img_base64 = base64.b64encode(img_jpg).decode('utf-8')

        object_label = msg.object_label
        room_label = msg.room_label
        if room_label == "":
            description = f"This is a {object_label}. {{ 'Obejct Type': {object_label}, 'Room Type': Unknown }}, Here is its image."
        else:        
            description = f"This is a {object_label} in the {room_label}. {{ 'Obejct Type': {object_label}, 'Room Type': {room_label} }}, Here is its image."

        instruction = f"Instruction: {self.instruction}, {{ 'Target Object': {self.target_object}, 'Room Condition': {self.room_condition if self.room_condition else 'None'}, 'Attribute Condition': {self.attribute_condition if self.attribute_condition else 'None'} , 'Spatial Condition': {self.spatial_condition if self.spatial_condition else 'None'} }} \n Determine whether the following object matches the target object described in the instruction."

        surrounding_description = " Here are some images of the surrounding environment from different viewpoints."

        class Result(BaseModel):
            reason: str
            is_target: bool
        try:
            # Process the target object with spatial query
            content_blocks = [
                {"type": "text", "text": instruction},
                {"type": "text", "text": description},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
            ]
            if len(viewpoint_img_bases) > 0:
                content_blocks.append({"type": "text", "text": surrounding_description})
            for vp_img_base in viewpoint_img_bases:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{vp_img_base}"
                    }
                })
            completion = self.vlm_model.beta.chat.completions.parse(
                model=self.object_type_vlm_model,
                messages=[{
                    "role": "system",
                    "content": self.target_object_spatial_prompt
                }, {
                    "role":
                        "user",
                    "content": content_blocks
                }],
                response_format=Result,
            )
            answer = completion.choices[0].message.parsed
            # print the answer
            self.get_logger().info(f"Received target object with spatial answer: {answer}")
            is_target = answer.is_target
            self.get_logger().info(f"Is target object: {is_target}")
            # Publish the target object with spatial answer
            answer_msg = TargetObject()
            answer_msg.header = msg.header
            answer_msg.object_id = msg.object_id
            answer_msg.img_path = msg.img_path
            answer_msg.object_label = msg.object_label
            answer_msg.room_label = msg.room_label
            answer_msg.is_target = is_target
            self.target_object_answer_publisher.publish(answer_msg)
            self.get_logger().info("Published target object with spatial answer")

            text = f"Object ID: {msg.object_id}\nObject Label: {object_label}\nRoom Label: {room_label}\nIs Target: {is_target}\nReason: {answer.reason}"
            self.publish_text_overlay(text)
            # save the image for debugging
            time_int = int(time.time())
            img_path = f"debug/target_object_spatial/{time_int}_{msg.object_id}_{object_label}_{is_target}.jpg"
            cv2.imwrite(img_path, img)
            for i, vp_img in enumerate(viewpoint_imgs):
                vp_img_path = f"debug/target_object_spatial/{time_int}_{msg.object_id}_{object_label}_{is_target}_viewpoint_{i}.jpg"
                cv2.imwrite(vp_img_path, vp_img)
            # save the answer to a text file for debugging
            answer_file_path = f"debug/target_object_spatial/{time_int}_{msg.object_id}_{object_label}_{is_target}.txt"
            with open(answer_file_path, 'w') as f:
                f.write(f"Instruction: {instruction}\nDescription: {description}\n")
                f.write(f"Is Target: {is_target}\nReason: {answer.reason}")
        except Exception as e:
            self.get_logger().error(f"Error processing target object with spatial query: {e}")

    def process_anchor_object_query(self, msg):
        """Handle anchor object query and publish answer"""
        img = np.load(msg.img_path)
        mask_path = msg.img_path.replace('.npy', '_mask.npy')
        if os.path.exists(mask_path):
            mask = np.load(mask_path, mmap_mode=None)
            mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
            # img = cv2.bitwise_and(img, img, mask=mask)

            # find contours of the mask and draw them
            mask_uint8 = mask.astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # draw the contours on the image for debugging
            cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        img_jpg = cv2.imencode('.jpg', img)[1]
        img_base64 = base64.b64encode(img_jpg).decode('utf-8')

        object_label = msg.object_label
        room_label = msg.room_label
        if room_label == "":
            description = f"This is a {object_label}. {{ 'Obejct Type': {object_label}, 'Room Type': Unknown }}, Here is its image."
        else:        
            description = f"This is a {object_label} in the {room_label}. {{ 'Obejct Type': {object_label}, 'Room Type': {room_label} }}, Here is its image."

        instruction = f"Instruction: {self.instruction_anchor}, {{ 'Target Object': {self.anchor_object}, 'Room Condition': {self.room_condition if self.room_condition else 'None'}, 'Attribute Condition': {self.attribute_condition_anchor if self.attribute_condition_anchor else 'None'} }} \n Determine whether the following object matches the target object described in the instruction."

        class Result(BaseModel):
            reason: str
            is_target: bool
        try:
            # Process the target object query
            completion = self.vlm_model.beta.chat.completions.parse(
                model=self.object_type_vlm_model,
                messages=[{
                    "role": "system",
                    "content": self.target_object_prompt
                }, {
                    "role":
                        "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "text", "text": description},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                    ]
                }],
                response_format=Result,
            )
            answer = completion.choices[0].message.parsed
            # print the answer
            self.get_logger().info(f"Received target object answer: {answer}")
            is_target = answer.is_target
            self.get_logger().info(f"Is target object: {is_target}")
            # Publish the target object answer
            answer_msg = TargetObject()
            answer_msg.header = msg.header
            answer_msg.object_id = msg.object_id
            answer_msg.img_path = msg.img_path
            answer_msg.object_label = msg.object_label
            answer_msg.room_label = msg.room_label
            answer_msg.is_target = is_target
            self.anchor_object_answer_publisher.publish(answer_msg)
            self.get_logger().info("Published target object answer")

            text = f"Anchor Object ID: {msg.object_id}\nAnchor Object Label: {object_label}\nRoom Label: {room_label}\nIs Anchor Object: {is_target}\nReason: {answer.reason}"
            self.publish_text_overlay(text)
            # save the image for debugging
            time_int = int(time.time())
            img_path = f"debug/anchor_object/{time_int}_{msg.object_id}_{object_label}_{is_target}.jpg"
            cv2.imwrite(img_path, img)
            # save the answer to a text file for debugging
            answer_file_path = f"debug/anchor_object/{time_int}_{msg.object_id}_{object_label}_{is_target}.txt"
            with open(answer_file_path, 'w') as f:
                f.write(f"Instruction: {instruction}\nDescription: {description}\n")
                f.write(f"Is Anchor Object: {is_target}\nReason: {answer.reason}")
        except Exception as e:
            self.get_logger().error(f"Error processing target object query: {e}")

    def vlm_node_callback(self):
        """Main loop to process queries"""
        # check if there are any room type queries
        if self.room_type_query_queue:
            latest_queries = {}
            # for each room_id, only keep the latest query
            while self.room_type_query_queue:
                item = self.room_type_query_queue.pop()  # 先出最新的
                room_id = item.room_id
                if room_id not in latest_queries or Time.from_msg(item.header.stamp) > Time.from_msg(latest_queries[room_id].header.stamp):
                    latest_queries[room_id] = item
            # using multithreading to process room type queries
            for room_id, query in latest_queries.items():
                self.get_logger().info(f"Processing room type query for room {room_id}")
                threading.Thread(target=self.process_room_type_query, args=(query,)).start()
        
        # check if there are any room early stop 1 queries
        if self.room_early_stop_1_query_queue:
            while self.room_early_stop_1_query_queue:
                item = self.room_early_stop_1_query_queue.popleft()
                self.get_logger().info(f"Processing room early stop 1 query for rooms {item.room_id_1} and {item.room_id_2}")
                threading.Thread(target=self.process_room_early_stop_1_query, args=(item,)).start()

        # check if there are any room navigation queries
        if self.room_navigation_query_queue:
            while self.room_navigation_query_queue:
                item = self.room_navigation_query_queue.popleft()
                self.get_logger().info(f"Processing room navigation query: {item.json}")
                threading.Thread(target=self.process_room_navigation_query, args=(item,)).start()
        
        # check if there are any target object queries
        if self.object_type_query_queue:
            while self.object_type_query_queue:
                item = self.object_type_query_queue.popleft()
                self.get_logger().info(f"Processing target object query for object ID: {item.object_id}")
                threading.Thread(target=self.process_object_type_query, args=(item,)).start()
    
        # check if there are any new instructions
        if self.instruction_queue:
            while self.instruction_queue:
                instruction = self.instruction_queue.popleft()
                self.get_logger().info(f"Processing new instruction: {instruction}")
                threading.Thread(target=self.process_instruction, args=(instruction,)).start()
        
        # process the target object query
        self.target_object_counter += 1
        # if self.target_object_counter % 10 == 0:
        if True:
            self.target_object_counter = 0
            if self.target_object_query_queue:
                latest_queries = {}
                while self.target_object_query_queue:
                    item = self.target_object_query_queue.pop()  # 先出最新的
                    object_id = item.object_id
                    if object_id not in latest_queries or Time.from_msg(item.header.stamp) > Time.from_msg(latest_queries[object_id].header.stamp):
                        latest_queries[object_id] = item
                for object_id, query in latest_queries.items():
                    self.get_logger().info(f"Processing target object query for object ID: {object_id}")
                    threading.Thread(target=self.process_target_object_query, args=(query,)).start()

        # process the target object with spatial query
        self.target_object_spatial_counter += 1
        if self.target_object_spatial_counter % 10 == 0:
            self.target_object_spatial_counter = 0
            if self.target_object_spatial_query_queue:
                latest_queries = {}
                while self.target_object_spatial_query_queue:
                    item = self.target_object_spatial_query_queue.pop()  # 先出最新的
                    object_id = item.object_id
                    if object_id not in latest_queries or Time.from_msg(item.header.stamp) > Time.from_msg(latest_queries[object_id].header.stamp):
                        latest_queries[object_id] = item
                for object_id, query in latest_queries.items():
                    self.get_logger().info(f"Processing target object with spatial query for object ID: {object_id}")
                    threading.Thread(target=self.process_target_object_spatial_query, args=(query,)).start()
        
        # process the anchor object query
        self.anchor_object_counter += 1
        if self.anchor_object_counter % 10 == 0:
            self.anchor_object_counter = 0
            if self.anchor_object_query_queue:
                latest_queries = {}
                while self.anchor_object_query_queue:
                    item = self.anchor_object_query_queue.pop()  # 先出最新的
                    object_id = item.object_id
                    if object_id not in latest_queries or Time.from_msg(item.header.stamp) > Time.from_msg(latest_queries[object_id].header.stamp):
                        latest_queries[object_id] = item
                for object_id, query in latest_queries.items():
                    self.get_logger().info(f"Processing anchor object query for object ID: {object_id}")
                    threading.Thread(target=self.process_anchor_object_query, args=(query,)).start()

        # self.publish_text_overlay("VLM Node is running...")
    
        
        

def main(args=None):
    rclpy.init(args=args)
    try:
        node = VLMNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("⏹️ Keyboard interrupt received")
    except Exception as e:
        print(f"❌ Failed to start VLM Node: {e}")
    finally:
        rclpy.shutdown()


# if __name__ == '__main__':
#     main()
