"""
Comprehensive tests for standard_e2e.data_structures.trajectory_data module.

This module provides thorough testing coverage for:
- TrajectoryComponent enum
- Helper functions (_to_numpy_1d_float32, _pad_numpy_1d, _validate_components_arg)
- Trajectory class (NumPy-based trajectory container)
- BatchedTrajectory class (PyTorch-based batched trajectory container)

The tests are organized into logical groups with clear documentation and cover:
- Basic functionality
- Edge cases and error conditions
- Integration scenarios
- Performance considerations
"""

import unittest

import numpy as np
import torch

from standard_e2e.data_structures.trajectory_data import (
    BatchedTrajectory,
    Trajectory,
    TrajectoryComponent,
    _pad_numpy_1d,
    _to_numpy_1d_float32,
    _validate_components_arg,
)


class TestTrajectoryComponent(unittest.TestCase):
    """Test cases for TrajectoryComponent enum."""

    def test_enum_string_behavior(self):
        """Test that TrajectoryComponent behaves correctly as string enum."""
        self.assertEqual(TrajectoryComponent.X, "x")
        self.assertEqual(TrajectoryComponent.Y, "y")
        self.assertTrue(isinstance(TrajectoryComponent.X, str))

    def test_enum_comparison_and_ordering(self):
        """Test enum comparison and ordering behavior."""
        # String comparisons should work
        self.assertEqual(TrajectoryComponent.X, "x")
        self.assertNotEqual(TrajectoryComponent.X, "y")

        # Enum comparisons should work
        self.assertEqual(TrajectoryComponent.X, TrajectoryComponent.X)
        self.assertNotEqual(TrajectoryComponent.X, TrajectoryComponent.Y)

    def test_enum_in_collections(self):
        """Test that enum works correctly in sets, dicts, and lists."""
        components_set = {TrajectoryComponent.X, TrajectoryComponent.Y}
        self.assertIn(TrajectoryComponent.X, components_set)
        self.assertNotIn(TrajectoryComponent.Z, components_set)

        components_dict = {
            TrajectoryComponent.X: "x_data",
            TrajectoryComponent.Y: "y_data",
        }
        self.assertEqual(components_dict[TrajectoryComponent.X], "x_data")


class TestHelperFunctions(unittest.TestCase):
    """Test cases for helper functions."""

    def test_to_numpy_1d_float32_numpy_arrays(self):
        """Test _to_numpy_1d_float32 with various numpy array inputs."""
        # Test 1D array (int)
        arr_1d_int = np.array([1, 2, 3])
        result = _to_numpy_1d_float32(arr_1d_int)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.float32)

        # Test 1D array (float64)
        arr_1d_float64 = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        result = _to_numpy_1d_float32(arr_1d_float64)
        expected = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.float32)

        # Test 2D array with shape (N, 1)
        arr_2d = np.array([[1], [2], [3]])
        result = _to_numpy_1d_float32(arr_2d)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.float32)

    def test_to_numpy_1d_float32_sequences(self):
        """Test _to_numpy_1d_float32 with lists and tuples."""
        # Test with list of ints
        list_data = [1, 2, 3]
        result = _to_numpy_1d_float32(list_data)
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Test with tuple of floats
        tuple_data = (1.5, 2.5, 3.5)
        result = _to_numpy_1d_float32(tuple_data)
        expected = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Test with mixed types in list
        mixed_data = [1, 2.5, 3]
        result = _to_numpy_1d_float32(mixed_data)
        expected = np.array([1.0, 2.5, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_to_numpy_1d_float32_edge_cases(self):
        """Test _to_numpy_1d_float32 with edge cases."""
        # Test empty array
        empty_array = np.array([])
        result = _to_numpy_1d_float32(empty_array)
        expected = np.array([], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Test single element
        single_element = [42.0]
        result = _to_numpy_1d_float32(single_element)
        expected = np.array([42.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_to_numpy_1d_float32_invalid_inputs(self):
        """Test _to_numpy_1d_float32 error handling."""
        # Test invalid type
        with self.assertRaises(TypeError):
            _to_numpy_1d_float32("invalid_string")

        with self.assertRaises(TypeError):
            _to_numpy_1d_float32(42)  # Single scalar

        with self.assertRaises(TypeError):
            _to_numpy_1d_float32(None)

        # Test invalid array shapes
        arr_3d = np.array([[[1, 2], [3, 4]]])
        with self.assertRaises(ValueError):
            _to_numpy_1d_float32(arr_3d)

        # Test 2D array with wrong shape
        arr_2d_wrong = np.array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            _to_numpy_1d_float32(arr_2d_wrong)

        # Test 2D array with shape (1, N) instead of (N, 1)
        arr_2d_transposed = np.array([[1, 2, 3]])
        with self.assertRaises(ValueError):
            _to_numpy_1d_float32(arr_2d_transposed)

    def test_pad_numpy_1d_basic(self):
        """Test _pad_numpy_1d basic functionality."""
        arr = np.array([1, 2, 3], dtype=np.float32)

        # Test right padding
        result = _pad_numpy_1d(arr, 5, "right")
        expected = np.array([1, 2, 3, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Test left padding
        result = _pad_numpy_1d(arr, 5, "left")
        expected = np.array([0, 0, 1, 2, 3], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_pad_numpy_1d_edge_cases(self):
        """Test _pad_numpy_1d edge cases."""
        arr = np.array([1, 2, 3], dtype=np.float32)

        # Test no padding needed
        result = _pad_numpy_1d(arr, 3, "right")
        np.testing.assert_array_equal(result, arr)

        # Test target length smaller than current (should return copy)
        result = _pad_numpy_1d(arr, 2, "right")
        np.testing.assert_array_equal(result, arr)

        # Test empty array padding
        empty_arr = np.array([], dtype=np.float32)
        result = _pad_numpy_1d(empty_arr, 3, "right")
        expected = np.array([0, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_pad_numpy_1d_invalid_side(self):
        """Test _pad_numpy_1d with invalid side parameter."""
        arr = np.array([1, 2, 3], dtype=np.float32)

        with self.assertRaises(ValueError):
            _pad_numpy_1d(arr, 5, "center")

        with self.assertRaises(ValueError):
            _pad_numpy_1d(arr, 5, "top")

    def test_validate_components_arg_single(self):
        """Test _validate_components_arg with single component."""
        result = _validate_components_arg(TrajectoryComponent.X)
        self.assertEqual(result, [TrajectoryComponent.X])

    def test_validate_components_arg_sequence(self):
        """Test _validate_components_arg with sequence of components."""
        components = [
            TrajectoryComponent.X,
            TrajectoryComponent.Y,
            TrajectoryComponent.Z,
        ]
        result = _validate_components_arg(components)
        self.assertEqual(result, components)

        # Test with tuple
        components_tuple = (TrajectoryComponent.X, TrajectoryComponent.Y)
        result = _validate_components_arg(components_tuple)
        self.assertEqual(result, list(components_tuple))

    def test_validate_components_arg_invalid(self):
        """Test _validate_components_arg error handling."""
        # Test invalid single type
        with self.assertRaises(TypeError):
            _validate_components_arg("invalid")

        with self.assertRaises(TypeError):
            _validate_components_arg(42)

        # Test empty sequence
        with self.assertRaises(TypeError):
            _validate_components_arg([])

        # Test invalid component in sequence
        with self.assertRaises(TypeError):
            _validate_components_arg([TrajectoryComponent.X, "invalid"])

        with self.assertRaises(TypeError):
            _validate_components_arg([TrajectoryComponent.X, None])


class TestTrajectoryCore(unittest.TestCase):
    """Test core Trajectory class functionality."""

    def test_initialization_empty(self):
        """Test initialization of empty trajectory."""
        traj = Trajectory()

        self.assertEqual(traj.length, 0)
        self.assertTrue(traj.isEmpty)
        self.assertEqual(traj.components(), [])
        self.assertIsNone(traj.score)
        self.assertEqual(len(traj), 0)

    def test_initialization_with_data(self):
        """Test initialization with data dictionary."""
        data = {TrajectoryComponent.X: [1, 2, 3], TrajectoryComponent.Y: [4, 5, 6]}
        traj = Trajectory(data=data)

        self.assertEqual(traj.length, 3)
        self.assertFalse(traj.isEmpty)
        self.assertIn(TrajectoryComponent.X, traj.components())
        self.assertIn(TrajectoryComponent.Y, traj.components())
        # IS_VALID should be auto-created
        self.assertIn(TrajectoryComponent.IS_VALID, traj.components())

    def test_initialization_with_score(self):
        """Test initialization with score."""
        traj = Trajectory(score=0.85)
        self.assertEqual(traj.score, 0.85)

        # Test with various score types
        traj_int = Trajectory(score=1)
        self.assertEqual(traj_int.score, 1.0)
        self.assertIsInstance(traj_int.score, float)

    def test_initialization_with_time_lattice_resampling(self):
        """Test initialization that triggers resampling."""
        data = {
            TrajectoryComponent.TIMESTAMP: [0.0, 1.0, 2.0],
            TrajectoryComponent.X: [0, 1, 2],
            TrajectoryComponent.Y: [0, 2, 4],
        }
        time_lattice = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

        traj = Trajectory(data=data, time_lattice=time_lattice)

        self.assertEqual(traj.length, 5)
        self.assertTrue(traj.has(TrajectoryComponent.TIMESTAMP))

        # Check that resampling occurred
        timestamps = traj.get(TrajectoryComponent.TIMESTAMP)
        np.testing.assert_array_equal(timestamps.flatten(), time_lattice)


class TestTrajectoryComponentManagement(unittest.TestCase):
    """Test component setting and getting functionality."""

    def test_set_single_component(self):
        """Test setting a single component."""
        traj = Trajectory()

        traj.set(TrajectoryComponent.X, [1.0, 2.0, 3.0])

        self.assertTrue(traj.has(TrajectoryComponent.X))
        self.assertEqual(traj.length, 3)
        self.assertFalse(traj.isEmpty)

    def test_set_multiple_components_set_many(self):
        """Test setting multiple components at once."""
        traj = Trajectory()

        data = {
            TrajectoryComponent.X: [1, 2, 3],
            TrajectoryComponent.Y: [4, 5, 6],
            TrajectoryComponent.SPEED: [10, 15, 20],
        }
        traj.set_many(data)

        self.assertEqual(traj.length, 3)
        for component in data.keys():
            self.assertTrue(traj.has(component))
        # IS_VALID should be auto-created
        self.assertTrue(traj.has(TrajectoryComponent.IS_VALID))

    def test_get_single_component_strict_mode(self):
        """Test getting single component in strict mode."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1.0, 2.0, 3.0])

        result = traj.get(TrajectoryComponent.X)
        expected = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (3, 1))

    def test_get_multiple_components_strict_mode(self):
        """Test getting multiple components in strict mode."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3])
        traj.set(TrajectoryComponent.Y, [4, 5, 6])

        result = traj.get([TrajectoryComponent.X, TrajectoryComponent.Y])
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (3, 2))

    def test_get_missing_component_strict_mode_raises(self):
        """Test that getting missing component in strict mode raises KeyError."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3])

        with self.assertRaises(KeyError) as cm:
            traj.get(TrajectoryComponent.Y, strict=True)

        self.assertIn("Missing component(s): Y", str(cm.exception))
        self.assertIn("Available: [", str(cm.exception))

    def test_get_missing_component_non_strict_mode_zeros(self):
        """Test getting missing component in non-strict mode returns zeros."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3])

        result = traj.get(TrajectoryComponent.Y, strict=False)
        expected = np.zeros((3, 1), dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

        # Test multiple components with some missing
        result = traj.get(
            [TrajectoryComponent.X, TrajectoryComponent.Y, TrajectoryComponent.Z],
            strict=False,
        )
        expected = np.array(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32
        )
        np.testing.assert_array_equal(result, expected)

    def test_has_component(self):
        """Test has() method."""
        traj = Trajectory()

        self.assertFalse(traj.has(TrajectoryComponent.X))

        traj.set(TrajectoryComponent.X, [1, 2, 3])
        self.assertTrue(traj.has(TrajectoryComponent.X))
        self.assertFalse(traj.has(TrajectoryComponent.Y))

    def test_components_list(self):
        """Test components() method returns correct list."""
        traj = Trajectory()
        self.assertEqual(traj.components(), [])

        traj.set(TrajectoryComponent.X, [1, 2, 3])
        components = traj.components()
        self.assertIn(TrajectoryComponent.X, components)
        self.assertIn(TrajectoryComponent.IS_VALID, components)  # Auto-created

        traj.set(TrajectoryComponent.Y, [4, 5, 6])
        components = traj.components()
        self.assertIn(TrajectoryComponent.Y, components)


class TestTrajectoryValidation(unittest.TestCase):
    """Test trajectory validation and error handling."""

    def test_length_mismatch_error(self):
        """Test that setting components with different lengths raises error."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3])

        with self.assertRaises(ValueError) as cm:
            traj.set(TrajectoryComponent.Y, [4, 5])  # Wrong length

        self.assertIn("Length mismatch", str(cm.exception))

    def test_invalid_component_types(self):
        """Test validation of component values."""
        traj = Trajectory()

        # These should work (various valid input types)
        traj.set(TrajectoryComponent.X, [1, 2, 3])  # list
        traj.set(TrajectoryComponent.Y, (4, 5, 6))  # tuple
        traj.set(TrajectoryComponent.Z, np.array([7, 8, 9]))  # numpy array

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        traj = Trajectory()

        # Setting empty data should work
        traj.set(TrajectoryComponent.X, [])
        self.assertEqual(traj.length, 0)
        self.assertTrue(traj.isEmpty)

    def test_auto_is_valid_creation(self):
        """Test automatic IS_VALID component creation."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3])

        # IS_VALID should be auto-created
        self.assertTrue(traj.has(TrajectoryComponent.IS_VALID))
        result = traj.get(TrajectoryComponent.IS_VALID)
        expected = np.ones((3, 1), dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_auto_is_valid_not_override_existing(self):
        """Test that auto IS_VALID doesn't override existing values."""
        traj = Trajectory()

        # Set IS_VALID first
        traj.set(TrajectoryComponent.IS_VALID, [1, 0, 1])
        original_is_valid = traj.get(TrajectoryComponent.IS_VALID).copy()

        # Add another component - should not change IS_VALID
        traj.set(TrajectoryComponent.X, [1, 2, 3])

        result_is_valid = traj.get(TrajectoryComponent.IS_VALID)
        np.testing.assert_array_equal(result_is_valid, original_is_valid)

    def test_is_valid_manual_override(self):
        """Test manual override of IS_VALID component."""
        data = {
            TrajectoryComponent.X: [1, 2, 3],
            TrajectoryComponent.Y: [4, 5, 6],
            TrajectoryComponent.IS_VALID: [1, 0, 1],  # Manual override
        }
        traj = Trajectory()
        traj.set_many(data)

        result = traj.get(TrajectoryComponent.IS_VALID)
        expected = np.array([[1.0], [0.0], [1.0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    # --- NEW timestamp ordering tests ---
    def test_timestamp_must_be_strictly_increasing_set(self):
        """Setting TIMESTAMP with non-increasing values should raise."""
        traj = Trajectory()
        with self.assertRaises(ValueError):
            traj.set(TrajectoryComponent.TIMESTAMP, [0.0, 0.0, 1.0])  # duplicate
        with self.assertRaises(ValueError):
            traj.set(TrajectoryComponent.TIMESTAMP, [0.0, -1.0, 2.0])  # decrease

    def test_timestamp_must_be_strictly_increasing_set_many(self):
        """set_many should validate TIMESTAMP ordering."""
        traj = Trajectory()
        with self.assertRaises(ValueError):
            traj.set_many(
                {
                    TrajectoryComponent.TIMESTAMP: [1.0, 1.0, 2.0],
                    TrajectoryComponent.X: [10, 20, 30],
                }
            )

    def test_resample_time_lattice_must_be_increasing(self):
        """Resampling with a non-increasing lattice should raise."""
        traj = Trajectory(
            {
                TrajectoryComponent.TIMESTAMP: [0.0, 1.0, 2.0],
                TrajectoryComponent.X: [0, 10, 20],
            }
        )
        with self.assertRaises(ValueError):
            traj.resample([0.0, 0.5, 0.5, 1.0])  # duplicate 0.5
        with self.assertRaises(ValueError):
            traj.resample([0.0, 0.7, 0.6, 1.0])  # decrease 0.7 -> 0.6

    def test_resample_validates_existing_timestamps(self):
        """Resample should also validate pre-existing stored timestamps (legacy)."""
        traj = Trajectory()
        # Bypass validation: directly set invalid timestamps internally (legacy data)
        traj._data[  # pylint: disable=protected-access
            TrajectoryComponent.TIMESTAMP
        ] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        traj.resample([0.0, 0.25, 0.5])
        assert np.array_equal(
            traj.get(TrajectoryComponent.TIMESTAMP),
            np.array([[0.0], [0.25], [0.5]], dtype=np.float32),
        )


class TestTrajectoryPaddingTrimming(unittest.TestCase):
    """Test trajectory padding and trimming operations."""

    def test_pad_right_side(self):
        """Test padding on the right side."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3])

        traj.pad(5, side="right")
        self.assertEqual(traj.length, 5)

        result = traj.get(TrajectoryComponent.X)
        expected = np.array([[1], [2], [3], [0], [0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_pad_left_side(self):
        """Test padding on the left side."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3])

        traj.pad(5, side="left")
        self.assertEqual(traj.length, 5)

        result = traj.get(TrajectoryComponent.X)
        expected = np.array([[0], [0], [1], [2], [3]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_pad_no_change_when_target_smaller(self):
        """Test that pad doesn't change trajectory when target is smaller."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3, 4, 5])

        original_data = traj.get(TrajectoryComponent.X).copy()
        traj.pad(3)  # Should not change anything

        np.testing.assert_array_equal(traj.get(TrajectoryComponent.X), original_data)
        self.assertEqual(traj.length, 5)

    def test_pad_creates_is_valid_if_missing(self):
        """Test that padding creates IS_VALID component if missing."""
        traj = Trajectory()
        # Manually set data without triggering auto IS_VALID creation
        # pylint: disable=protected-access
        traj.set(TrajectoryComponent.X, [1, 2, 3])

        self.assertTrue(traj.has(TrajectoryComponent.IS_VALID))

        traj.pad(5)

        self.assertTrue(traj.has(TrajectoryComponent.IS_VALID))
        result = traj.get(TrajectoryComponent.IS_VALID)
        expected = np.array([[1], [1], [1], [0], [0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_trim_right_side(self):
        """Test trimming from the right side."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3, 4, 5])

        traj.trim(3, side="right")
        self.assertEqual(traj.length, 3)

        result = traj.get(TrajectoryComponent.X)
        expected = np.array([[1], [2], [3]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_trim_left_side(self):
        """Test trimming from the left side."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3, 4, 5])

        traj.trim(3, side="left")
        self.assertEqual(traj.length, 3)

        result = traj.get(TrajectoryComponent.X)
        expected = np.array([[3], [4], [5]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_trim_no_change_when_target_larger(self):
        """Test that trim doesn't change trajectory when target is larger."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3])

        original_data = traj.get(TrajectoryComponent.X).copy()
        traj.trim(5)  # Should not change anything

        np.testing.assert_array_equal(traj.get(TrajectoryComponent.X), original_data)
        self.assertEqual(traj.length, 3)

    def test_pad_or_trim_padding_case(self):
        """Test pad_or_trim when padding is needed."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3])

        traj.pad_or_trim(5)
        self.assertEqual(traj.length, 5)

        result = traj.get(TrajectoryComponent.X)
        expected = np.array([[1], [2], [3], [0], [0]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_pad_or_trim_trimming_case(self):
        """Test pad_or_trim when trimming is needed."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3, 4, 5])

        traj.pad_or_trim(3)
        self.assertEqual(traj.length, 3)

        result = traj.get(TrajectoryComponent.X)
        expected = np.array([[1], [2], [3]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_pad_or_trim_no_change(self):
        """Test pad_or_trim when no change is needed."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3])

        original_data = traj.get(TrajectoryComponent.X).copy()
        traj.pad_or_trim(3)  # Same length

        np.testing.assert_array_equal(traj.get(TrajectoryComponent.X), original_data)
        self.assertEqual(traj.length, 3)

    def test_invalid_side_parameter_raises(self):
        """Test that invalid side parameter raises ValueError."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [1, 2, 3])

        with self.assertRaises(ValueError):
            traj.pad(5, side="center")

        with self.assertRaises(ValueError):
            traj.trim(2, side="middle")


class TestTrajectoryResampling(unittest.TestCase):
    """Test trajectory resampling functionality."""

    def test_resample_basic_interpolation(self):
        """Test basic linear interpolation resampling."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.TIMESTAMP, [0.0, 1.0, 2.0])
        traj.set(TrajectoryComponent.X, [0, 10, 20])
        traj.set(TrajectoryComponent.Y, [0, 5, 10])

        new_times = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        traj = traj.resample(new_times)

        self.assertEqual(traj.length, 5)

        # Check timestamps
        timestamps = traj.get(TrajectoryComponent.TIMESTAMP)
        np.testing.assert_array_equal(timestamps.flatten(), new_times)

        # Check interpolated values
        x_values = traj.get(TrajectoryComponent.X).flatten()
        expected_x = np.array([0, 5, 10, 15, 20], dtype=np.float32)
        np.testing.assert_array_equal(x_values, expected_x)
        y_values = traj.get(TrajectoryComponent.Y).flatten()
        expected_y = np.array([0, 2.5, 5, 7.5, 10], dtype=np.float32)
        np.testing.assert_array_equal(y_values, expected_y)

    def test_resample_with_is_valid_component(self):
        """Test resampling with IS_VALID component using only valid timestamps."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.TIMESTAMP, [0.0, 1.0, 2.0, 3.0, 4.0])
        traj.set(TrajectoryComponent.X, [0, 10, -4096, 20, 30])
        traj.set(TrajectoryComponent.IS_VALID, [1, 1, 0, 1, 1])  # Invalid at t=2

        new_times = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
        traj = traj.resample(new_times)

        # Check that IS_VALID is binary
        is_valid = traj.get(TrajectoryComponent.IS_VALID).flatten()
        for val in is_valid:
            self.assertIn(val, [0.0, 1.0])

        # Valid intervals should remain disjoint: [0.0,1.0] and [3.0,4.0]
        # New times inside gaps stay invalid
        expected_valid = np.array(
            [
                0,  # -1
                1,  # 0.0
                1,  # 0.5 (between 0 and1)
                1,  # 1.0
                0,  # 1.5 gap
                0,  # 2.0 gap (original invalid)
                0,  # 2.5 gap
                1,  # 3.0
                1,  # 3.5
                1,  # 4.0
                0,  # 5.0 outside
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(is_valid, expected_valid)

        # Check that interpolation uses only valid points
        x_values = traj.get(TrajectoryComponent.X).flatten()
        # Interpolated values at gap times should be zeroed out
        expected_x = np.array(
            [
                0.0,  # -1 outside
                0.0,  # 0.0
                5.0,  # 0.5
                10.0,  # 1.0
                0.0,  # 1.5 gap -> zero
                0.0,  # 2.0 gap -> zero
                0.0,  # 2.5 gap -> zero
                20.0,  # 3.0
                25.0,  # 3.5
                30.0,  # 4.0
                0.0,  # 5.0 outside
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(x_values, expected_x)

        # Values outside valid range should be zero
        self.assertEqual(x_values[0], 0.0)  # t=-1.0 (outside range)
        self.assertEqual(x_values[10], 0.0)  # t=5.0 (outside range)

    def test_resample_with_is_valid_component_2(self):
        """Test resampling with IS_VALID component using only valid timestamps."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.TIMESTAMP, [0.0, 1.0, 2.0, 3.0, 4.0])
        traj.set(TrajectoryComponent.X, [0, 10, 5, 20, 30])
        traj.set(TrajectoryComponent.IS_VALID, [1, 1, 1, 1, 1])

        new_times = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
        traj = traj.resample(new_times)

        # Check that IS_VALID is binary
        is_valid = traj.get(TrajectoryComponent.IS_VALID).flatten()
        for val in is_valid:
            self.assertIn(val, [0.0, 1.0])

        expected_valid = np.array(
            [
                0,  # -1
                1,  # 0.0
                1,  # 0.5
                1,  # 1.0
                1,  # 1.5
                1,  # 2.0
                1,  # 2.5
                1,  # 3.0
                1,  # 3.5
                1,  # 4.0
                0,  # 5.0
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(is_valid, expected_valid)

        # Check that interpolation uses only valid points
        x_values = traj.get(TrajectoryComponent.X).flatten()
        # Interpolated values at gap times should be zeroed out
        expected_x = np.array(
            [
                0.0,  # -1
                0.0,  # 0.0
                5.0,  # 0.5
                10.0,  # 1.0
                7.5,  # 1.5
                5.0,  # 2.0
                12.5,  # 2.5
                20.0,  # 3.0
                25.0,  # 3.5
                30.0,  # 4.0
                0.0,  # 5.0
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(x_values, expected_x)

        # Values outside valid range should be zero
        self.assertEqual(x_values[0], 0.0)  # t=-1.0 (outside range)
        self.assertEqual(x_values[10], 0.0)  # t=5.0 (outside range)

    def test_resample_with_is_valid_component_3(self):
        """Test resampling with IS_VALID component using only valid timestamps."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.TIMESTAMP, [0.0, 1.0, 2.0, 3.0, 4.0])
        traj.set(TrajectoryComponent.X, [0, 10, 5, 20, 30])
        traj.set(TrajectoryComponent.IS_VALID, [0, 1, 1, 1, 0])

        new_times = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
        traj = traj.resample(new_times)

        # Check that IS_VALID is binary
        is_valid = traj.get(TrajectoryComponent.IS_VALID).flatten()
        for val in is_valid:
            self.assertIn(val, [0.0, 1.0])

        expected_valid = np.array(
            [
                0,  # -1
                0,  # 0.0
                0,  # 0.5
                1,  # 1.0
                1,  # 1.5
                1,  # 2.0
                1,  # 2.5
                1,  # 3.0
                0,  # 3.5
                0,  # 4.0
                0,  # 5.0
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(is_valid, expected_valid)

        # Check that interpolation uses only valid points
        x_values = traj.get(TrajectoryComponent.X).flatten()
        # Interpolated values at gap times should be zeroed out
        expected_x = np.array(
            [
                0.0,  # -1
                0.0,  # 0.0
                0.0,  # 0.5
                10.0,  # 1.0
                7.5,  # 1.5
                5.0,  # 2.0
                12.5,  # 2.5
                20.0,  # 3.0
                0.0,  # 3.5
                0.0,  # 4.0
                0.0,  # 5.0
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(x_values, expected_x)

    def test_resample_with_is_valid_component_4(self):
        """Test resampling with IS_VALID component using only valid timestamps."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.TIMESTAMP, [0.0, 1.0, 2.0, 3.0, 4.0])
        traj.set(TrajectoryComponent.X, [0, 10, 5, 20, 30])
        traj.set(TrajectoryComponent.IS_VALID, [0, 1, 0, 1, 0])

        new_times = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
        traj = traj.resample(new_times)

        # Check that IS_VALID is binary
        is_valid = traj.get(TrajectoryComponent.IS_VALID).flatten()
        for val in is_valid:
            self.assertIn(val, [0.0, 1.0])

        expected_valid = np.array(
            [
                0,  # -1
                0,  # 0.0
                0,  # 0.5
                1,  # 1.0
                0,  # 1.5
                0,  # 2.0
                0,  # 2.5
                1,  # 3.0
                0,  # 3.5
                0,  # 4.0
                0,  # 5.0
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(is_valid, expected_valid)

        # Check that interpolation uses only valid points
        x_values = traj.get(TrajectoryComponent.X).flatten()
        # Interpolated values at gap times should be zeroed out
        expected_x = np.array(
            [
                0.0,  # -1
                0.0,  # 0.0
                0.0,  # 0.5
                10.0,  # 1.0
                0.0,  # 1.5
                0.0,  # 2.0
                0.0,  # 2.5
                20.0,  # 3.0
                0.0,  # 3.5
                0.0,  # 4.0
                0.0,  # 5.0
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(x_values, expected_x)

    def test_resample_with_all_invalid_points(self):
        """Test resampling when all original IS_VALID points are False."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.TIMESTAMP, [1.0, 2.0, 3.0])
        traj.set(TrajectoryComponent.X, [10, 20, 30])
        traj.set(TrajectoryComponent.IS_VALID, [0, 0, 0])  # All invalid

        new_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        traj.resample(new_times)

        # All should be zero since no original points were valid
        x_values = traj.get(TrajectoryComponent.X).flatten()
        is_valid = traj.get(TrajectoryComponent.IS_VALID).flatten()

        expected_x = np.zeros(5, dtype=np.float32)
        expected_valid = np.zeros(5, dtype=np.float32)

        np.testing.assert_array_equal(x_values, expected_x)
        np.testing.assert_array_equal(is_valid, expected_valid)

    def test_resample_without_timestamp_raises(self):
        """Test that resampling without TIMESTAMP component raises error."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.X, [0, 10, 20])

        new_times = np.array([0.0, 0.5, 1.0])

        with self.assertRaises(ValueError) as cm:
            traj.resample(new_times)

        self.assertIn(
            "Cannot resample trajectory without TIMESTAMP component", str(cm.exception)
        )

    def test_resample_single_point_trajectory(self):
        """Test resampling trajectory with single point."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.TIMESTAMP, [1.0])
        traj.set(TrajectoryComponent.X, [10])
        traj.set(TrajectoryComponent.Y, [20])

        new_times = np.array([0.0, 1.0, 2.0])
        traj.resample(new_times)

        self.assertEqual(traj.length, 3)

        # For single point, non-timestamp components should be filled with the value
        x_values = traj.get(TrajectoryComponent.X).flatten()
        expected_x = np.array([0, 10, 0], dtype=np.float32)
        np.testing.assert_array_equal(x_values, expected_x)

        # IS_VALID should be False for times outside the original point
        is_valid = traj.get(TrajectoryComponent.IS_VALID).flatten()
        expected_valid = np.array(
            [0, 1, 0], dtype=np.float32
        )  # Only original time is valid
        np.testing.assert_array_equal(is_valid, expected_valid)

    def test_resample_with_internal_invalid_gap(self):
        """Gap inside validity should remain invalid after resampling."""
        traj = Trajectory()
        # Original timestamps 0..8
        original_times = list(range(9))
        traj.set(TrajectoryComponent.TIMESTAMP, original_times)
        # X values with a deliberate jump so interpolation across the gap is testable.
        # Use sentinel -4096 for invalid region to ensure it is ignored.
        x_values = [0, 5, 20, -4096, -4096, -4096, 60, 70, 80]
        traj.set(TrajectoryComponent.X, x_values)
        # Valid mask has internal gap (indices 3,4,5 invalid)
        is_valid_pattern = [1, 1, 1, 0, 0, 0, 1, 1, 1]
        traj.set(TrajectoryComponent.IS_VALID, is_valid_pattern)

        # New lattice extends beyond original range and includes original times
        new_times = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32)
        traj = traj.resample(new_times)

        # Expected validity: two disjoint intervals [0,2] and [6,8]
        expected_is_valid = np.array(
            [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype=np.float32
        )
        resampled_is_valid = traj.get(TrajectoryComponent.IS_VALID).flatten()
        np.testing.assert_array_equal(resampled_is_valid, expected_is_valid)

        # Interpolation should bridge the gap using only valid points (2 -> 6)
        # Valid times & values used: (0,0), (1,5), (2,20), (6,60), (7,70), (8,80)
        # Linear interpolation between (2,20) and (6,60) gives 30,40,50 at times 3,4,5.
        expected_x = np.array(
            [
                0.0,  # -1 outside
                0.0,  # 0
                5.0,  # 1
                20.0,  # 2
                0.0,  # 3 gap -> zero
                0.0,  # 4 gap -> zero
                0.0,  # 5 gap -> zero
                60.0,  # 6
                70.0,  # 7
                80.0,  # 8
                0.0,  # 9 outside
            ],
            dtype=np.float32,
        )
        resampled_x = traj.get(TrajectoryComponent.X).flatten()
        np.testing.assert_array_equal(resampled_x, expected_x)
        # Ensure sentinel values are gone
        self.assertNotIn(-4096, resampled_x.tolist())

    def test_resample_with_disjoint_intervals_example(self):
        """User-specified example: preserve two valid intervals [1,3] and [9,11]."""
        traj = Trajectory()
        times = [1, 3, 5, 7, 9, 11]
        traj.set(TrajectoryComponent.TIMESTAMP, times)
        # Values (invalid points will be ignored):
        x_vals = [10, 30, 50, 70, 90, 110]
        traj.set(TrajectoryComponent.X, x_vals)
        is_valid = [1, 1, 0, 0, 1, 1]
        traj.set(TrajectoryComponent.IS_VALID, is_valid)

        new_times = np.array(list(range(13)), dtype=np.float32)  # 0..12
        traj = traj.resample(new_times)

        expected_is_valid = np.array(
            [
                0,  # 0 outside first interval
                1,  # 1
                1,  # 2 inside [1,3]
                1,  # 3
                0,  # 4 gap
                0,  # 5 original invalid
                0,  # 6 gap
                0,  # 7 original invalid
                0,  # 8 gap
                1,  # 9
                1,  # 10 inside [9,11]
                1,  # 11
                0,  # 12 outside
            ],
            dtype=np.float32,
        )
        got_is_valid = traj.get(TrajectoryComponent.IS_VALID).flatten()
        np.testing.assert_array_equal(got_is_valid, expected_is_valid)

        # Expected X: interpolate within each valid interval only
        # Valid points used: (1,10),(3,30) and (9,90),(11,110)
        expected_x = np.array(
            [
                0.0,  # 0 outside
                10.0,  # 1
                20.0,  # 2 between 1 and 3
                30.0,  # 3
                0.0,  # 4 gap
                0.0,  # 5 invalid
                0.0,  # 6 gap
                0.0,  # 7 invalid
                0.0,  # 8 gap
                90.0,  # 9
                100.0,  # 10 between 9 and 11
                110.0,  # 11
                0.0,  # 12 outside
            ],
            dtype=np.float32,
        )
        got_x = traj.get(TrajectoryComponent.X).flatten()
        np.testing.assert_array_equal(got_x, expected_x)

    def test_resample_empty_trajectory(self):
        """Test resampling empty trajectory."""
        traj = Trajectory()
        new_times = np.array([0.0, 1.0, 2.0])

        traj.resample(new_times)

        self.assertEqual(traj.length, 3)
        self.assertTrue(traj.has(TrajectoryComponent.TIMESTAMP))
        self.assertTrue(traj.has(TrajectoryComponent.IS_VALID))

        # All should be invalid for empty original trajectory
        is_valid = traj.get(TrajectoryComponent.IS_VALID).flatten()
        np.testing.assert_array_equal(is_valid, np.zeros(3, dtype=np.float32))

    def test_resample_extrapolation_behavior(self):
        """Test resampling behavior outside original time range (sets to zero)."""
        traj = Trajectory()
        traj.set(TrajectoryComponent.TIMESTAMP, [1.0, 2.0, 3.0])
        traj.set(TrajectoryComponent.X, [10, 20, 30])

        # Include times outside the original range
        new_times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        traj = traj.resample(new_times)

        # Check that values outside original time range are set to zero
        x_values = traj.get(TrajectoryComponent.X).flatten()
        self.assertEqual(x_values[0], 0.0)  # Outside range -> zero
        self.assertEqual(x_values[1], 10.0)  # Within range -> interpolated
        self.assertEqual(x_values[2], 20.0)  # Within range -> interpolated
        self.assertEqual(x_values[3], 30.0)  # Within range -> interpolated
        self.assertEqual(x_values[-1], 0.0)  # Outside range -> zero

        # Check validity - should be invalid outside original range
        is_valid = traj.get(TrajectoryComponent.IS_VALID).flatten()
        expected_valid = np.array([0, 1, 1, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(is_valid, expected_valid)


class TestTrajectoryUtilities(unittest.TestCase):
    """Test trajectory utility methods and properties."""

    def test_length_property(self):
        """Test length property."""
        traj = Trajectory()
        self.assertEqual(traj.length, 0)

        traj.set(TrajectoryComponent.X, [1, 2, 3, 4])
        self.assertEqual(traj.length, 4)

    def test_isEmpty_property(self):
        """Test isEmpty property."""
        traj = Trajectory()
        self.assertTrue(traj.isEmpty)

        traj.set(TrajectoryComponent.X, [1, 2, 3])
        self.assertFalse(traj.isEmpty)

    def test_score_property(self):
        """Test score property."""
        traj = Trajectory()
        self.assertIsNone(traj.score)

        traj.score = 0.75
        self.assertEqual(traj.score, 0.75)

    def test_repr_string(self):
        """Test string representation."""
        traj = Trajectory()
        repr_str = repr(traj)
        self.assertIn("Trajectory", repr_str)
        self.assertIn("N=0", repr_str)

        traj.set(TrajectoryComponent.X, [1, 2, 3])
        traj.score = 0.9
        repr_str = repr(traj)
        self.assertIn("N=3", repr_str)
        self.assertIn("score=0.9", repr_str)

    def test_len_dunder_method(self):
        """Test __len__ method."""
        traj = Trajectory()
        self.assertEqual(len(traj), 0)

        traj.set(TrajectoryComponent.X, [1, 2, 3, 4, 5])
        self.assertEqual(len(traj), 5)


class TestBatchedTrajectoryCore(unittest.TestCase):
    """Test core BatchedTrajectory functionality."""

    def setUp(self):
        """Set up test trajectories for batched tests."""
        # Create test trajectories with different characteristics
        self.traj1 = Trajectory(
            {
                TrajectoryComponent.X: [1, 2, 3],
                TrajectoryComponent.Y: [4, 5, 6],
                TrajectoryComponent.SPEED: [10, 15, 20],
            },
            score=0.8,
        )

        self.traj2 = Trajectory(
            {
                TrajectoryComponent.X: [7, 8],
                TrajectoryComponent.Y: [9, 10],
                TrajectoryComponent.SPEED: [25, 30],
            },
            score=0.9,
        )

        self.traj3 = Trajectory(
            {
                TrajectoryComponent.X: [11, 12, 13, 14],
                TrajectoryComponent.Y: [15, 16, 17, 18],
                TrajectoryComponent.SPEED: [35, 40, 45, 50],
            },
            score=0.7,
        )

        self.empty_traj = Trajectory()

    def test_initialization_basic(self):
        """Test basic BatchedTrajectory initialization."""
        batch = BatchedTrajectory([self.traj1, self.traj2])

        self.assertEqual(batch.batch_size, 2)
        self.assertEqual(batch.length, 3)  # Max length from traj1
        self.assertEqual(batch.device, torch.device("cpu"))
        self.assertIsInstance(batch.scores, torch.Tensor)
        self.assertIsInstance(batch.is_empty_mask, torch.Tensor)

    def test_initialization_with_device(self):
        """Test initialization with specified device."""
        device = torch.device("cpu")
        batch = BatchedTrajectory([self.traj1, self.traj2], device=device)
        self.assertEqual(batch.device, device)

        # Test CUDA if available
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            batch_cuda = BatchedTrajectory([self.traj1, self.traj2], device=cuda_device)
            self.assertEqual(batch_cuda.device, cuda_device)

    def test_initialization_empty_list_raises(self):
        """Test that empty trajectory list raises ValueError."""
        with self.assertRaises(ValueError) as cm:
            BatchedTrajectory([])
        self.assertIn("non-empty list", str(cm.exception))

    def test_initialization_with_empty_trajectories(self):
        """Test initialization with some empty trajectories."""
        batch = BatchedTrajectory([self.traj1, self.empty_traj, self.traj2])

        self.assertEqual(batch.batch_size, 3)
        # Check empty mask
        empty_mask = batch.is_empty_mask
        expected_mask = torch.tensor([False, True, False])
        torch.testing.assert_close(empty_mask, expected_mask)

    def test_initialization_strict_mode_component_mismatch_raises(self):
        """Test strict mode raises error for component mismatch."""
        # Create trajectory with different components
        traj_different = Trajectory({TrajectoryComponent.HEADING: [0, 1, 2]})

        with self.assertRaises(ValueError) as cm:
            BatchedTrajectory([self.traj1, traj_different], strict=True)
        self.assertIn("component mismatch", str(cm.exception))

    def test_initialization_non_strict_mode_different_components(self):
        """Test non-strict mode handles different components."""
        traj_different = Trajectory({TrajectoryComponent.HEADING: [0, 1, 2]})

        # Should work in non-strict mode
        batch = BatchedTrajectory([self.traj1, traj_different], strict=False)
        self.assertEqual(batch.batch_size, 2)

        # Should have union of all components
        components = batch.components()
        self.assertIn(TrajectoryComponent.X, components)
        self.assertIn(TrajectoryComponent.HEADING, components)

    def test_properties(self):
        """Test BatchedTrajectory properties."""
        batch = BatchedTrajectory([self.traj1, self.traj2, self.traj3])

        # Test batch_size
        self.assertEqual(batch.batch_size, 3)

        # Test length (should be max length)
        self.assertEqual(batch.length, 4)  # traj3 has length 4

        # Test scores
        scores = batch.scores
        self.assertEqual(scores.shape, (3,))
        self.assertAlmostEqual(scores[0].item(), 0.8)
        self.assertAlmostEqual(scores[1].item(), 0.9)
        self.assertAlmostEqual(scores[2].item(), 0.7)

        # Test is_empty_mask
        empty_mask = batch.is_empty_mask
        expected_mask = torch.tensor([False, False, False])
        torch.testing.assert_close(empty_mask, expected_mask)

    def test_components_and_has_methods(self):
        """Test components() and has() methods."""
        batch = BatchedTrajectory([self.traj1, self.traj2])

        components = batch.components()
        expected_components = {
            TrajectoryComponent.X,
            TrajectoryComponent.Y,
            TrajectoryComponent.SPEED,
            TrajectoryComponent.IS_VALID,
        }
        self.assertEqual(set(components), expected_components)

        # Test has method
        self.assertTrue(batch.has(TrajectoryComponent.X))
        self.assertTrue(batch.has(TrajectoryComponent.Y))
        self.assertTrue(batch.has(TrajectoryComponent.SPEED))
        self.assertTrue(batch.has(TrajectoryComponent.IS_VALID))
        self.assertFalse(batch.has(TrajectoryComponent.HEADING))

    def test_get_single_component(self):
        """Test getting single component from batch."""
        batch = BatchedTrajectory([self.traj1, self.traj2])

        result = batch.get(TrajectoryComponent.X)
        self.assertEqual(result.shape, (2, 3, 1))  # (batch, seq, features)
        self.assertEqual(result.dtype, torch.float32)

        # Check values
        expected_x1 = torch.tensor([1, 2, 3], dtype=torch.float32)
        expected_x2 = torch.tensor([7, 8, 0], dtype=torch.float32)  # Padded
        torch.testing.assert_close(result[0, :, 0], expected_x1)
        torch.testing.assert_close(result[1, :, 0], expected_x2)

    def test_get_multiple_components(self):
        """Test getting multiple components from batch."""
        batch = BatchedTrajectory([self.traj1, self.traj2])

        result = batch.get([TrajectoryComponent.X, TrajectoryComponent.Y])
        self.assertEqual(result.shape, (2, 3, 2))  # (batch, seq, features)

        # Check that components are concatenated correctly
        x_values = result[:, :, 0]
        y_values = result[:, :, 1]

        expected_x1 = torch.tensor([1, 2, 3], dtype=torch.float32)
        expected_y1 = torch.tensor([4, 5, 6], dtype=torch.float32)
        torch.testing.assert_close(x_values[0], expected_x1)
        torch.testing.assert_close(y_values[0], expected_y1)

    def test_get_missing_component_strict_raises(self):
        """Test getting missing component in strict mode raises error."""
        batch = BatchedTrajectory([self.traj1, self.traj2])

        with self.assertRaises(KeyError) as cm:
            batch.get(TrajectoryComponent.HEADING)
        self.assertIn("Missing component(s)", str(cm.exception))

    def test_get_missing_component_non_strict_zeros(self):
        """Test getting missing component in non-strict mode returns zeros."""
        batch = BatchedTrajectory([self.traj1, self.traj2])

        result = batch.get(TrajectoryComponent.HEADING, strict=False)
        expected_shape = (2, 3, 1)  # (batch, seq, 1)
        self.assertEqual(result.shape, expected_shape)
        torch.testing.assert_close(result, torch.zeros(expected_shape))

    def test_device_operations_to_method(self):
        """Test to() device movement method."""
        batch = BatchedTrajectory([self.traj1, self.traj2])

        # Test moving to same device (should return self)
        same_device_batch = batch.to(torch.device("cpu"))
        self.assertIs(same_device_batch, batch)

        # Test moving to different device if CUDA available
        if torch.cuda.is_available():
            cuda_batch = batch.to(torch.device("cuda"))
            self.assertEqual(cuda_batch.device.type, "cuda")
            self.assertIs(cuda_batch, batch)  # Should return self

    def test_device_operations_cuda_method(self):
        """Test cuda() device movement method."""
        batch = BatchedTrajectory([self.traj1, self.traj2])

        if torch.cuda.is_available():
            cuda_batch = batch.cuda()
            self.assertEqual(cuda_batch.device.type, "cuda")
            self.assertIs(cuda_batch, batch)  # Should return self

    def test_pad_operations(self):
        """Test batch-level padding operations."""
        batch = BatchedTrajectory([self.traj1, self.traj2])

        # Test padding (should increase length)
        batch.pad(5)
        self.assertEqual(batch.length, 5)

        # Test that data shapes are updated
        result = batch.get(TrajectoryComponent.X)
        self.assertEqual(result.shape, (2, 5, 1))

    def test_trim_operations(self):
        """Test batch-level trimming operations."""
        batch = BatchedTrajectory([self.traj1, self.traj2, self.traj3])
        self.assertEqual(batch.length, 4)  # Max from traj3

        # Test trimming
        batch.trim(2)
        self.assertEqual(batch.length, 2)

        # Test that data is actually trimmed
        result = batch.get(TrajectoryComponent.X)
        self.assertEqual(result.shape, (3, 2, 1))

    def test_pad_or_trim_operations(self):
        """Test pad_or_trim combined operations."""
        # Test padding case
        batch_copy1 = BatchedTrajectory([self.traj1, self.traj2])
        batch_copy1.pad_or_trim(5)
        self.assertEqual(batch_copy1.length, 5)

        # Test trimming case
        batch_copy2 = BatchedTrajectory([self.traj1, self.traj2, self.traj3])
        batch_copy2.pad_or_trim(2)
        self.assertEqual(batch_copy2.length, 2)

        # Test no change case
        batch_copy3 = BatchedTrajectory([self.traj1, self.traj2])
        original_len = batch_copy3.length
        batch_copy3.pad_or_trim(original_len)
        self.assertEqual(batch_copy3.length, original_len)


class TestBatchedTrajectoryEdgeCases(unittest.TestCase):
    """Test edge cases for BatchedTrajectory."""

    def test_mixed_length_trajectories(self):
        """Test handling of trajectories with very different lengths."""
        short_traj = Trajectory({TrajectoryComponent.X: [1]})
        medium_traj = Trajectory({TrajectoryComponent.X: [1, 2, 3, 4, 5]})
        long_traj = Trajectory({TrajectoryComponent.X: list(range(20))})

        batch = BatchedTrajectory([short_traj, medium_traj, long_traj])
        self.assertEqual(batch.batch_size, 3)
        self.assertEqual(batch.length, 20)  # Max length

        # Test that shorter trajectories are padded
        result = batch.get(TrajectoryComponent.X)
        self.assertEqual(result.shape, (3, 20, 1))

    def test_all_empty_trajectories(self):
        """Test batch with all empty trajectories."""
        empty1 = Trajectory()
        empty2 = Trajectory()
        empty3 = Trajectory()

        batch = BatchedTrajectory([empty1, empty2, empty3])
        self.assertEqual(batch.batch_size, 3)
        self.assertEqual(batch.length, 0)

        # All should be marked as empty
        empty_mask = batch.is_empty_mask
        expected_mask = torch.tensor([True, True, True])
        torch.testing.assert_close(empty_mask, expected_mask)

    def test_single_trajectory_batch(self):
        """Test batch with single trajectory."""
        single_traj = Trajectory(
            {TrajectoryComponent.X: [1, 2, 3], TrajectoryComponent.Y: [4, 5, 6]}
        )

        batch = BatchedTrajectory([single_traj])
        self.assertEqual(batch.batch_size, 1)
        self.assertEqual(batch.length, 3)

        result = batch.get(TrajectoryComponent.X)
        self.assertEqual(result.shape, (1, 3, 1))

    def test_trajectories_with_nan_scores(self):
        """Test handling of trajectories with NaN or None scores."""
        traj_no_score = Trajectory({TrajectoryComponent.X: [1, 2, 3]})
        traj_with_score = Trajectory({TrajectoryComponent.X: [4, 5, 6]}, score=0.8)

        batch = BatchedTrajectory([traj_no_score, traj_with_score])
        scores = batch.scores

        self.assertTrue(torch.isnan(scores[0]))  # None score -> NaN
        self.assertAlmostEqual(scores[1].item(), 0.8)

    def test_repr_string(self):
        """Test BatchedTrajectory string representation."""
        traj1 = Trajectory({TrajectoryComponent.X: [1, 2, 3]})
        traj2 = Trajectory({TrajectoryComponent.Y: [4, 5, 6]})

        batch = BatchedTrajectory([traj1, traj2], strict=False)
        repr_str = repr(batch)

        self.assertIn("BatchedTrajectory", repr_str)
        self.assertIn("batch_size=2", repr_str)
        self.assertIn("sequence_length=3", repr_str)
        self.assertIn("device=", repr_str)


class TestBatchedTrajectoryResample(unittest.TestCase):
    """Tests for BatchedTrajectory.resample functionality."""

    def _make_equal_length_trajs(self):
        # Two trajectories same length so no padding alters TIMESTAMP
        traj1 = Trajectory(
            {
                TrajectoryComponent.TIMESTAMP: [0.0, 1.0, 2.0],
                TrajectoryComponent.X: [0.0, 10.0, 20.0],
            },
            score=0.5,
        )
        traj2 = Trajectory(
            {
                TrajectoryComponent.TIMESTAMP: [0.0, 1.0, 2.0],
                TrajectoryComponent.X: [5.0, 15.0, 25.0],
            },
            score=0.6,
        )
        return traj1, traj2

    def test_resample_basic_new_batch_values(self):
        """Resample returns new batch with expected interpolation and length."""
        traj1, traj2 = self._make_equal_length_trajs()
        batch = BatchedTrajectory([traj1, traj2])

        new_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        new_batch = batch.resample(new_times)

        # Original batch unchanged
        self.assertEqual(batch.length, 3)
        # New batch length equals new lattice
        self.assertEqual(new_batch.length, 5)
        self.assertIsNot(batch, new_batch)

        # Components present
        self.assertTrue(new_batch.has(TrajectoryComponent.TIMESTAMP))
        self.assertTrue(new_batch.has(TrajectoryComponent.X))
        self.assertTrue(new_batch.has(TrajectoryComponent.IS_VALID))

        x_values = new_batch.get(TrajectoryComponent.X)[:, :, 0]
        # Expected linear interpolation
        expected_traj1 = torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0])
        expected_traj2 = torch.tensor([5.0, 10.0, 15.0, 20.0, 25.0])
        torch.testing.assert_close(x_values[0], expected_traj1)
        torch.testing.assert_close(x_values[1], expected_traj2)

        # All points valid inside original span
        is_valid = new_batch.get(TrajectoryComponent.IS_VALID)[:, :, 0]
        torch.testing.assert_close(is_valid, torch.ones_like(is_valid))

    def test_resample_with_empty_trajectory(self):
        """Empty trajectory becomes length of lattice with invalid mask zeros."""
        traj1, traj2 = self._make_equal_length_trajs()
        empty_traj = Trajectory()
        assert empty_traj.isEmpty
        batch = BatchedTrajectory([traj1, empty_traj, traj2])
        new_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        new_batch = batch.resample(new_times)

        self.assertEqual(new_batch.length, 5)
        # Middle trajectory was empty: after resample timestamps present, IS_VALID zeros
        is_valid = new_batch.get(TrajectoryComponent.IS_VALID)[1, :, 0]
        torch.testing.assert_close(is_valid, torch.zeros(5))
        timestamps = new_batch.get(TrajectoryComponent.TIMESTAMP)[1, :, 0]
        torch.testing.assert_close(
            timestamps, torch.tensor(new_times, dtype=torch.float32)
        )

        # Non-empty trajectories still valid across range
        valid_first = new_batch.get(TrajectoryComponent.IS_VALID)[0, :, 0]
        torch.testing.assert_close(valid_first, torch.ones(5))


class TestTrajectoryIntegration(unittest.TestCase):
    """Test integration scenarios between Trajectory and BatchedTrajectory."""

    def test_trajectory_to_batched_conversion(self):
        """Test converting individual trajectories to batched format."""
        trajectories = []
        for i in range(5):
            traj = Trajectory(
                {
                    TrajectoryComponent.X: list(range(i + 1, i + 4)),
                    TrajectoryComponent.Y: list(range(i + 10, i + 13)),
                    TrajectoryComponent.SPEED: [20 + i * 5] * 3,
                },
                score=0.1 * i,
            )
            trajectories.append(traj)

        batch = BatchedTrajectory(trajectories)

        # Test that all data is preserved
        self.assertEqual(batch.batch_size, 5)
        for i in range(5):
            self.assertAlmostEqual(batch.scores[i].item(), 0.1 * i)

    def test_roundtrip_consistency(self):
        """Test that data remains consistent through operations."""
        original_traj = Trajectory(
            {
                TrajectoryComponent.X: [1, 2, 3, 4, 5],
                TrajectoryComponent.Y: [10, 20, 30, 40, 50],
                TrajectoryComponent.SPEED: [100, 110, 120, 130, 140],
            },
            score=0.95,
        )

        # Create batch
        batch = BatchedTrajectory([original_traj])

        # Get data back
        x_data = batch.get(TrajectoryComponent.X)
        y_data = batch.get(TrajectoryComponent.Y)
        speed_data = batch.get(TrajectoryComponent.SPEED)

        # Verify data integrity
        expected_x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
        expected_y = torch.tensor([[10], [20], [30], [40], [50]], dtype=torch.float32)
        expected_speed = torch.tensor(
            [[100], [110], [120], [130], [140]], dtype=torch.float32
        )

        torch.testing.assert_close(x_data[0], expected_x)
        torch.testing.assert_close(y_data[0], expected_y)
        torch.testing.assert_close(speed_data[0], expected_speed)

    def test_large_batch_performance(self):
        """Test performance with larger batches."""
        # Create a moderately large batch for testing
        trajectories = []
        for _ in range(50):
            traj = Trajectory(
                {
                    TrajectoryComponent.X: np.random.randn(10).tolist(),
                    TrajectoryComponent.Y: np.random.randn(10).tolist(),
                    TrajectoryComponent.SPEED: (np.random.rand(10) * 50).tolist(),
                },
                score=np.random.rand(),
            )
            trajectories.append(traj)

        # This should complete without errors
        batch = BatchedTrajectory(trajectories)
        self.assertEqual(batch.batch_size, 50)

        # Test getting data
        result = batch.get([TrajectoryComponent.X, TrajectoryComponent.Y])
        self.assertEqual(result.shape, (50, 10, 2))


if __name__ == "__main__":
    # Configure test runner for better output
    unittest.main(verbosity=2, buffer=True)
