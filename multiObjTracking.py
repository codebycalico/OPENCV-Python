import cv2

# list of the different types of trackers
TrackerDictionary = { 'csrt': cv2.TrackerCSRT_create,
                      'kcf': cv2.TrackerCSRT_create,
                      'boosting': cv2.TrackerBoosting_create,
                      'mil': cv2.TrackerMIL_create,
                      'tld': cv2.TrackerTLD_create,
                      'medianflow': cv2.TrackerMedianFlow_create,
                      'mosse': cv2.TrackerMOSSE_create }

# initialize
csrtTracker = TrackerDictionary['csrt']()

# create capture object (video feed)
cap = cv2.VideoCapture(0)

while True:
    # read frame from capture object
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    bb = cv2.selectROI('frame', frame)

    csrtTracker.init(frame, bb)