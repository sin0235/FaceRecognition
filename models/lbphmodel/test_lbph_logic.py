"""
Script kiểm tra logic LBPH sau khi sửa chữa.
Verify rằng các functions hoạt động đúng và metrics nhất quán.
"""
import sys
import os
import numpy as np
import cv2

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.lbphmodel.train_lbph import train_lbph_model
from models.lbphmodel.evaluate_lbph import evaluate_lbph
from models.lbphmodel.threshold_lbph import find_optimal_threshold


def create_dummy_data(n_classes=5, n_samples_per_class=10, img_size=(100, 100)):
    """Tạo dummy data để test."""
    faces = []
    labels = []
    
    for class_id in range(n_classes):
        for _ in range(n_samples_per_class):
            # Tạo ảnh random với pattern khác nhau cho mỗi class
            img = np.random.randint(0, 255, img_size, dtype=np.uint8)
            # Thêm chut pattern để phân biệt classes
            img[class_id*10:(class_id+1)*10, :] = 255
            
            faces.append(img)
            labels.append(class_id)
    
    return faces, np.array(labels, dtype=np.int32)


def test_train_function():
    """Test train_lbph_model function."""
    print("\n" + "="*60)
    print("TEST 1: train_lbph_model()")
    print("="*60)
    
    # Tạo dummy training data
    train_faces, train_labels = create_dummy_data(n_classes=3, n_samples_per_class=5)
    
    print(f"Training data: {len(train_faces)} samples, {len(set(train_labels))} classes")
    
    # Train model
    model = train_lbph_model(train_faces, train_labels)
    
    print("✓ Model trained successfully")
    print(f"✓ Model type: {type(model)}")
    
    return model, train_faces, train_labels


def test_evaluate_function(model, faces, labels):
    """Test evaluate_lbph function."""
    print("\n" + "="*60)
    print("TEST 2: evaluate_lbph()")
    print("="*60)
    
    threshold = 50
    accuracy, coverage, used, confidences = evaluate_lbph(
        model, faces, labels, threshold
    )
    
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {accuracy:.4f} (should be in range [0, 1])")
    print(f"Coverage: {coverage:.4f} (should be in range [0, 1])")
    print(f"Used samples: {used}/{len(labels)}")
    print(f"Confidences shape: {confidences.shape}")
    
    # Verify metrics
    assert 0 <= accuracy <= 1, "Accuracy phải trong khoảng [0, 1]"
    assert 0 <= coverage <= 1, "Coverage phải trong khoảng [0, 1]"
    assert used <= len(labels), "Used không thể lớn hơn total samples"
    assert len(confidences) == len(labels), "Confidences phải có length = số samples"
    
    print("✓ All metrics are in valid ranges")
    

def test_threshold_function(model, faces, labels):
    """Test find_optimal_threshold function."""
    print("\n" + "="*60)
    print("TEST 3: find_optimal_threshold()")
    print("="*60)
    
    best_thr, history = find_optimal_threshold(
        model, faces, labels, 
        min_coverage=0.3,
        threshold_range=range(40, 101, 10)
    )
    
    print(f"Best threshold: {best_thr}")
    print(f"Search performed over {len(history)} thresholds")
    
    if history:
        print("\nMetrics history:")
        print(f"{'Thr':<6} {'Acc':<8} {'Cov':<8} {'Used':<6} {'Score':<8}")
        print("-" * 50)
        for entry in history:
            print(f"{entry['threshold']:<6} "
                  f"{entry['accuracy']:<8.3f} "
                  f"{entry['coverage']:<8.3f} "
                  f"{entry['used']:<6} "
                  f"{entry['score']:<8.4f}")
    
    # Verify
    assert best_thr is not None, "Best threshold không được None"
    assert isinstance(history, list), "History phải là list"
    
    print("✓ Threshold finding works correctly")
    

def test_consistency():
    """Test tính nhất quán của metrics."""
    print("\n" + "="*60)
    print("TEST 4: Consistency Check")
    print("="*60)
    
    # Tạo data
    faces, labels = create_dummy_data(n_classes=5, n_samples_per_class=8)
    
    # Train
    model = train_lbph_model(faces, labels)
    
    # Evaluate với nhiều thresholds
    thresholds = [40, 50, 60, 70, 80]
    
    print("\nComparing metrics across thresholds:")
    print(f"{'Thr':<6} {'Acc':<8} {'Cov':<8} {'Used':<6}")
    print("-" * 40)
    
    prev_coverage = 0
    for thr in thresholds:
        acc, cov, used, _ = evaluate_lbph(model, faces, labels, thr)
        print(f"{thr:<6} {acc:<8.3f} {cov:<8.3f} {used:<6}")
        
        # Coverage phải tăng khi threshold tăng (accept nhiều hơn)
        assert cov >= prev_coverage, f"Coverage phải tăng khi threshold tăng"
        prev_coverage = cov
    
    print("✓ Coverage increases monotonically with threshold")
    

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LBPH LOGIC VERIFICATION TESTS")
    print("="*60)
    
    try:
        # Test 1: Training
        model, train_faces, train_labels = test_train_function()
        
        # Test 2: Evaluation
        test_evaluate_function(model, train_faces, train_labels)
        
        # Test 3: Threshold finding
        test_threshold_function(model, train_faces, train_labels)
        
        # Test 4: Consistency
        test_consistency()
        
        print("\n" + "="*60)
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
