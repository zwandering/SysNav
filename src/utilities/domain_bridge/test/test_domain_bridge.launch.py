# Copyright 2021, Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import os
import time
import unittest

from launch import LaunchDescription
from launch.actions import ExecuteProcess
import launch_testing
from launch_testing.actions import ReadyToTest
import launch_testing.asserts
import launch_testing.tools
from launch_testing.util import KeepAliveProc


def generate_test_description():
    return LaunchDescription([
        KeepAliveProc(),
        ReadyToTest()
    ])


class TestDomainBridge(unittest.TestCase):

    @classmethod
    def setUpClass(
        cls,
        launch_service,
        proc_info,
        proc_output
    ):

        @contextlib.contextmanager
        def launch_domain_bridge(self, arguments):
            executable = os.path.join(os.getcwd(), 'domain_bridge')
            if os.name == 'nt':
                executable += '.exe'
            cmd = [executable] + arguments
            domain_bridge_proc = ExecuteProcess(
                cmd=cmd,
                output='screen',
            )
            with launch_testing.tools.launch_process(
                launch_service, domain_bridge_proc, proc_info, proc_output
            ) as launch_process:
                yield launch_process

        cls.launch_domain_bridge = launch_domain_bridge

        cls.example_yaml_path = os.path.join(
            os.path.dirname(__file__), '..', 'examples', 'example_bridge_config.yaml')

    def test_help(self):
        with self.launch_domain_bridge(arguments=['--help']) as command:
            assert command.wait_for_shutdown(timeout=3)
        assert command.exit_code == launch_testing.asserts.EXIT_OK
        assert launch_testing.tools.expect_output(
            expected_lines=[
                '    --help, -h               Print this help message.',
            ],
            text=command.output,
            strict=False
        )

    def test_bridge_invalid_file(self):
        path_to_yaml = 'file_should_NOT_exist'
        with self.launch_domain_bridge(arguments=[path_to_yaml]) as command:
            assert command.wait_for_shutdown(timeout=3)
        assert command.exit_code != launch_testing.asserts.EXIT_OK
        assert launch_testing.tools.expect_output(
            expected_lines=[
                "  what():  error parsing the file 'file_should_NOT_exist': file does not exist",
            ],
            text=command.output,
            strict=False
        )

    def test_bridge_valid_file(self, proc_info):
        with self.launch_domain_bridge(arguments=[self.example_yaml_path]) as command:
            # Let command run for a bit
            time.sleep(1)
        proc_info.assertWaitForShutdown(process=command.target_process_action, timeout=3)
        assert command.exit_code == launch_testing.asserts.EXIT_OK

    def test_bridge_valid_from_domain(self, proc_info):
        with self.launch_domain_bridge(
            arguments=['--from', '42', self.example_yaml_path]
        ) as command:
            # Let command run for a bit
            time.sleep(1)
        proc_info.assertWaitForShutdown(process=command.target_process_action, timeout=3)
        assert command.exit_code == launch_testing.asserts.EXIT_OK

    def test_bridge_valid_to_domain(self, proc_info):
        with self.launch_domain_bridge(arguments=['--to', '0', self.example_yaml_path]) as command:
            # Let command run for a bit
            time.sleep(1)
        proc_info.assertWaitForShutdown(process=command.target_process_action, timeout=3)
        assert command.exit_code == launch_testing.asserts.EXIT_OK

    def test_bridge_invalid_from_domain(self, proc_info):
        with self.launch_domain_bridge(
            arguments=['--from', '42.0', self.example_yaml_path]
        ) as command:
            assert command.wait_for_shutdown(timeout=3)
        assert command.exit_code != launch_testing.asserts.EXIT_OK
        assert launch_testing.tools.expect_output(
            expected_lines=[
                "error: Failed to parse FROM_DOMAIN_ID '42.0'",
            ],
            text=command.output,
            strict=False
        )

    def test_bridge_invalid_to_domain(self, proc_info):
        with self.launch_domain_bridge(
            arguments=['--to', 'not_a_num', self.example_yaml_path]
        ) as command:
            assert command.wait_for_shutdown(timeout=3)
        assert command.exit_code != launch_testing.asserts.EXIT_OK
        assert launch_testing.tools.expect_output(
            expected_lines=[
                "error: Failed to parse TO_DOMAIN_ID 'not_a_num'",
            ],
            text=command.output,
            strict=False
        )
