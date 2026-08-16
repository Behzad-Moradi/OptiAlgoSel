import pytest
import numpy as np
from API.services.doe_validator import validate_doe

NUM_SAMPLE_POINTS = 2500
PROB_DIM = 10


################################################################################
################################################################################

def create_doe(num_sample_points=2500, prob_dim=10):
    
    doe = np.random.uniform(-5, 5, size=(num_sample_points, prob_dim + 1))
    lb = -5 * np.ones(prob_dim)
    ub = 5 * np.ones(prob_dim)

    return doe, lb, ub

@pytest.fixture
def doe_res():
    return create_doe
################################################################################

@pytest.fixture
def mock_conn(mocker):
    conn = mocker.MagicMock()

    cursor = conn.cursor.return_value
    cursor.__enter__.return_value = cursor
    cursor.fetchone.return_value = (2500, 10)

    return conn
################################################################################
################################################################################

def test_sampling_set_configuration_not_found(mocker, doe_res):
    
    mock_conn = mocker.MagicMock()

    mock_cursor = mock_conn.cursor.return_value
    mock_cursor.__enter__.return_value = mock_cursor
    mock_cursor.fetchone.return_value = None
    
    doe, lb, ub = doe_res()
    with pytest.raises(ValueError, match="Sampling-set configuration was not found."):
        validate_doe(doe, lb, ub, mock_conn)
    
    mock_conn.cursor.return_value.execute.assert_called_once_with("SELECT num_points, problem_dim FROM sampling_sets")

################################################################################

def test_valid_doe(mock_conn, doe_res):

    doe, lb, ub = doe_res()

    assert validate_doe(doe, lb, ub, mock_conn) is None
    
################################################################################

@pytest.mark.parametrize(
    "invalid_input, expected_message",
    [
        ("doe", "DOE must be a NumPy array."),
        ("lb", "Lower bound vector must be a NumPy array."),
        ("ub", "Upper bound vector must be a NumPy array."),
    ]
)
def test_inputs_must_be_numpy_arrays(mock_conn, doe_res, invalid_input, expected_message):
    
    doe, lb, ub = doe_res()
    if invalid_input == "doe":
        doe = doe.tolist()
    elif invalid_input == "lb":
        lb = lb.tolist()
    else:
        ub = ub.tolist()

    with pytest.raises(TypeError, match=expected_message):
        validate_doe(doe, lb, ub, mock_conn)

################################################################################
        
def test_doe_shape(mock_conn, doe_res):
    
    doe, lb, ub = doe_res(2000, 10)
    with pytest.raises(ValueError, match="The number of points in the DOE does not match the number of required sampling points."):
        validate_doe(doe, lb, ub, mock_conn)

################################################################################
        
def test_doe_dim(mock_conn, doe_res):

    doe, lb, ub = doe_res(2500, 20)
    with pytest.raises(ValueError, match="The number of dimensions in the DOE does not match the required problem dimension."):
        validate_doe(doe, lb, ub, mock_conn)

################################################################################

@pytest.mark.parametrize(
    "value, expected_message",
    [
        ("lb", "The lower bound vector does not match the required problem dimension."),
        ("ub", "The upper bound vector does not match the required problem dimension."),
    ]
)
def test_doe_dim(mock_conn, doe_res, value, expected_message):
    
    doe, lb, ub = doe_res()
    if value == "lb":
        lb = -5*np.ones(20)
    else:
        ub = +5*np.ones(20)
    with pytest.raises(ValueError, match=expected_message):
        validate_doe(doe, lb, ub, mock_conn)

################################################################################

@pytest.mark.parametrize(
    "value, expected_message",
    [
        (np.inf, "Variable 1 in the DOE contains non-finite values."),
        (np.nan, "Variable 1 in the DOE contains non-finite values."),
    ]
)
def test_doe_infinite_values(mock_conn, doe_res, value, expected_message):  
    
    doe, lb, ub = doe_res()
    doe[:, 0] = value
    with pytest.raises(ValueError, match=expected_message):
        validate_doe(doe, lb, ub, mock_conn)

################################################################################

@pytest.mark.parametrize(
    "value, expected_message",
    [
        (-10, "Variable 1 in the DOE is out of bounds."),
        (10, "Variable 1 in the DOE is out of bounds."),
    ]
)
def test_doe_out_of_bounds(mock_conn, doe_res, value, expected_message):

    doe, lb, ub = doe_res()
    doe[:, 0] = value
    with pytest.raises(ValueError, match=expected_message):
        validate_doe(doe, lb, ub, mock_conn)

################################################################################

def test_infinite_obj_values(mock_conn, doe_res):
    
    doe, lb, ub = doe_res()
    doe[:, -1] = np.inf
    with pytest.raises(ValueError, match="The objective values column in the DOE contains non-finite values."):
        validate_doe(doe, lb, ub, mock_conn)
