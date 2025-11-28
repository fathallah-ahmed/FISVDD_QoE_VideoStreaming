import pytest
import numpy as np
from fisvdd import fisvdd

@pytest.fixture
def sample_data():
    # Create a simple 2D dataset
    np.random.seed(42)
    return np.random.rand(10, 2)

def test_fisvdd_initialization(sample_data):
    model = fisvdd(sample_data, sigma=0.5)
    assert model.sigma == 0.5
    assert model.sv.shape == (1, 2)  # Initialized with first point
    assert model.alpha.shape == (1,)
    assert model.score == 1.0

def test_fisvdd_training(sample_data):
    model = fisvdd(sample_data, sigma=0.5)
    model.find_sv()
    
    # Check if support vectors were updated
    assert len(model.sv) > 0
    assert len(model.alpha) == len(model.sv)
    # Alpha should sum to approximately 1
    assert np.isclose(np.sum(model.alpha), 1.0)

def test_score_fcn(sample_data):
    model = fisvdd(sample_data, sigma=0.5)
    model.find_sv()
    
    new_point = np.array([[0.5, 0.5]])
    score, sim_vec = model.score_fcn(new_point)
    
    assert isinstance(score, float)
    assert sim_vec.shape == (len(model.sv),)

def test_expand_shrink(sample_data):
    model = fisvdd(sample_data, sigma=0.1) # Small sigma to force more SVs
    
    # Manually expand
    new_sv = np.array([[0.9, 0.9]])
    score, sim_vec = model.score_fcn(new_sv)
    model.expand(new_sv, sim_vec)
    
    assert len(model.sv) == 2
    
    # Force shrink by setting a negative alpha (artificial test)
    model.alpha[0] = -0.1
    backup = model.shrink()
    
    assert len(backup) > 0
    assert len(model.sv) < 2


