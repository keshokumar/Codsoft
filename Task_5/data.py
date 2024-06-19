import cv2
import os

video=cv2.VideoCapture(0)

facedetect=cv2.CascadeClassifier('K:\Internship\Codsoft\Task_5\haarcascade_frontalface_default.xml')

count=0

nameID=str(input("Enter Your Name: ")).lower()

base_path=r'k:\Internship\Codsoft\Task_5'
path=os.path.join(base_path,'images/'+nameID)

if os.path.exists(path):
    print("Name Already Taken")
    nameID = str(input("Enter Your Name Again: ")).lower()
    path = os.path.join(base_path, 'images', nameID)
if not os.path.exists(path):
    os.makedirs(path)
	
print("Press 'q' to stop capturing images.")

while True:
	ret,frame=video.read()
	faces=facedetect.detectMultiScale(frame,1.3, 5)
	for x,y,w,h in faces:
		count=count+1
		image_path = os.path.join(path, f'{count}.jpg')
		print(f"Creating image:{image_path}")
       
		cv2.imwrite(image_path, frame[y:y+h,x:x+w])
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
	cv2.imshow("WindowFrame", frame)
	
	key = cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		print("Manaual stop triggering")
		break

	if count>500:
		break
video.release()
cv2.destroyAllWindows()

print("Image capture complete")