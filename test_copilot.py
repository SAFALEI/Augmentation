def normalize_angle(angle):
    """
    Normalize an angle to the range of [-180, 180) degrees.
    """
    while angle <= -180:
        angle += 360
    while angle > 180:
        angle -= 360
    return angle

def test_normalize_angle():
    """
    Test the normalize_angle function with various test cases.
    """
    # Test case 1: Angle within the range of -180 to 180 degrees
    assert normalize_angle(45) == 45
    assert normalize_angle(-45) == -45

    # Test case 2: Angle greater than 180 degrees
    assert normalize_angle(190) == -170
    assert normalize_angle(360) == 0
    assert normalize_angle(450) == 90

    # Test case 3: Angle less than -180 degrees
    assert normalize_angle(-190) == 170
    assert normalize_angle(-360) == 0
    assert normalize_angle(-450) == -90

    # Test case 4: Angle exactly at the boundaries
    assert normalize_angle(180) == 180
    assert normalize_angle(-180) == -180

    print("All tests passed!")

# Run the test function
test_normalize_angle()

