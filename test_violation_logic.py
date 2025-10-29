#!/usr/bin/env python3
"""
Test script để debug violation counting logic
"""

# Simulate the data structure
violation_data = {
    'motor_violations': 0,
    'car_violations': 0,
    'violation_cooldown': {}
}

def test_violation_logic():
    """Test logic đếm vi phạm"""
    print("🧪 Testing violation counting logic...")
    
    # Simulate video frames
    w = 1920  # Video width
    lane_boundary = w // 2  # 960
    grid_size = 400
    frame_count = 1
    
    # Test case 1: Xe máy (cls=1) vi phạm vào làn ô tô
    print("\n=== TEST CASE 1: Xe máy vi phạm ===")
    
    # Xe máy ở vị trí vi phạm (center_x > lane_boundary)
    cls = 1
    center_x = 1200  # > 960 (vi phạm)
    center_y = 500
    
    grid_x = center_x // grid_size  # 1200 // 400 = 3
    grid_y = center_y // grid_size  # 500 // 400 = 1
    
    violation_key = f"MOTOR_{grid_x}_{grid_y}_{cls}"  # "MOTOR_3_1_1"
    
    print(f"Xe máy: center_x={center_x}, boundary={lane_boundary}, key={violation_key}")
    
    # Lần 1: Vi phạm mới
    if violation_key not in violation_data['violation_cooldown']:
        old_count = violation_data['motor_violations']
        violation_data['motor_violations'] += 1
        violation_data['violation_cooldown'][violation_key] = frame_count
        print(f"✅ Lần 1: {old_count} -> {violation_data['motor_violations']}")
    
    # Lần 2: Same key - should skip
    frame_count += 1
    if violation_key not in violation_data['violation_cooldown']:
        violation_data['motor_violations'] += 1
        print(f"❌ Lần 2: Incremented (BAD)")
    else:
        print(f"✅ Lần 2: Skipped (GOOD)")
    
    # Test case 2: Ô tô vi phạm
    print("\n=== TEST CASE 2: Ô tô vi phạm ===")
    
    cls = 0  # Car class
    center_x = 400  # < 960 (vi phạm)  
    center_y = 600
    
    grid_x = center_x // grid_size  # 400 // 400 = 1
    grid_y = center_y // grid_size  # 600 // 400 = 1
    
    violation_key = f"CAR_{grid_x}_{grid_y}_{cls}"  # "CAR_1_1_0"
    
    print(f"Ô tô: center_x={center_x}, boundary={lane_boundary}, key={violation_key}")
    
    # Lần 1: Vi phạm mới
    if violation_key not in violation_data['violation_cooldown']:
        old_count = violation_data['car_violations']
        violation_data['car_violations'] += 1
        violation_data['violation_cooldown'][violation_key] = frame_count
        print(f"✅ Lần 1: {old_count} -> {violation_data['car_violations']}")
    
    # Test multiple cars in same grid
    print("\n=== TEST CASE 3: Nhiều ô tô cùng grid ===")
    
    # Car 1: cls=0, same position -> same key -> skip
    cls = 0
    center_x = 450  # Still in same grid
    center_y = 650  # Still in same grid
    
    grid_x = center_x // grid_size  # 450 // 400 = 1
    grid_y = center_y // grid_size  # 650 // 400 = 1
    
    violation_key = f"CAR_{grid_x}_{grid_y}_{cls}"  # "CAR_1_1_0" - SAME KEY!
    
    print(f"Ô tô 2: center_x={center_x}, key={violation_key}")
    
    if violation_key not in violation_data['violation_cooldown']:
        violation_data['car_violations'] += 1
        print(f"❌ Car 2: Incremented (BAD - same key)")
    else:
        print(f"✅ Car 2: Skipped (GOOD - same key)")
    
    # Car 3: Different class -> different key -> should count
    cls = 3  # Different class
    center_x = 450  # Same position 
    center_y = 650  # Same position
    
    grid_x = center_x // grid_size  
    grid_y = center_y // grid_size
    
    violation_key = f"CAR_{grid_x}_{grid_y}_{cls}"  # "CAR_1_1_3" - DIFFERENT KEY!
    
    print(f"Ô tô khác loại: center_x={center_x}, key={violation_key}")
    
    if violation_key not in violation_data['violation_cooldown']:
        old_count = violation_data['car_violations']
        violation_data['car_violations'] += 1
        violation_data['violation_cooldown'][violation_key] = frame_count
        print(f"✅ Car 3 (different class): {old_count} -> {violation_data['car_violations']}")
    
    print("\n=== FINAL RESULTS ===")
    print(f"Motor violations: {violation_data['motor_violations']}")
    print(f"Car violations: {violation_data['car_violations']}")
    print(f"Total tracked keys: {len(violation_data['violation_cooldown'])}")
    print(f"Keys: {list(violation_data['violation_cooldown'].keys())}")
    
    print("\n🎯 Expected: Motor=1, Car=2 (nếu logic đúng)")
    
    if violation_data['motor_violations'] == 1 and violation_data['car_violations'] == 2:
        print("✅ LOGIC ĐÚNG!")
    else:
        print("❌ LOGIC SAI!")

if __name__ == "__main__":
    test_violation_logic()