from ai_robotics.vision.multi_tracker import MultiObjectTracker



def test_new_object_get_id():
    tracker  = MultiObjectTracker()
    objects = tracker.update([(10, 20), (30, 40)])
    assert 1 in objects
    assert 2 in objects
def test_object_keep_ids():
    tracker =MultiObjectTracker()
    tracker.update([
        (100,100),
        (300,300)
    ])

    objects = tracker.update([

        (110,105),
        (295,305)
    ])

    assert objects[1] == (110,105)
    assert objects[2] == (295,305)

    

