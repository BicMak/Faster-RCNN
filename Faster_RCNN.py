# Let's import the necessary libs
import torch
import torchvision
import collections
import cv2
from torchvision.models.detection import faster_rcnn
import torchvision.models as models


class feature_extractor(torch.nn.Module):
    def __init__(self,model):
        super(feature_extractor,self).__init__()
        self.module = model

    def forward(self,x):
        return self.module(x)
    

class FasterRCNN(torch.nn.Module):
    def __init__(self,backbone,region_proposer,roi_pooler,mlp_head,final_layer):
        super(FasterRCNN,self).__init__()
        self.fruit_classes = {0:'banana',1:'Jackfruit',2:'Mango',3:'Litchi',4:'HogPlum',5:'Papaya',6:'Grapes',7:'Apple',8:'Orange',9:'Guava'}
        self.backbone = backbone
        self.region_proposer = region_proposer
        self.roi_pooler = roi_pooler
        self.mlp_head = mlp_head
        self.final_layer = final_layer
        
    def forward(self,x,targets):
        features = self.backbone(x)
        image_size = x.shape[-2:]
        proposals, proposal_losses = self.region_proposer(features, image_size, targets)
        box_features = self.roi_pooler(features, proposals, image_size)
        box_features = self.mlp_head(box_features)
        class_logits, box_regression = self.final_layer(box_features)
        return class_logits, box_regression, proposals, proposal_losses
    
    def image_preprocessor(self,image,Means,Stds):
        image_transformer=torchvision.transforms.Compose([
            torchvision.transforms.Resize((800,800)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(Means,Stds)
        ])
        return image_transformer(image)
    
    def get_prediction(self,image_path):
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image = self.image_preprocessor(original_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image = image.unsqueeze(0)  # Add batch dimension

        self.eval()
        with torch.no_grad():
            class_logits, box_regression, proposals, proposal_losses = self.forward(image, None)
            
        pred_class = [self.fruit_classes[i] for i in class_logits.argmax(dim=1).tolist()]
        pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in box_regression]
        pred_score = proposals[0] 
        pred_t = [i for i, score in enumerate(pred_score) if score > 0.5]
        result_boxes = [pred_boxes[i] for i in pred_t]
        result_classes = [pred_class[i] for i in pred_t]
        return result_boxes, result_classes
    
    def define_optimizer(self,lr):
        self.optimizaer = torch.optim.Adam(self.parameters(), lr = lr, weight_decay= 0.00005, betas=(0.9, 0.999))
    
    def detect_object(self,image_path,boxes,classes):
        prediction_box = [boxes, classes]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(prediction_box[0])):
            cv2.rectangle(img, prediction_box[0][i][0], prediction_box[0][i][1], (0, 255, 0), 2)
            cv2.putText(img, prediction_box[1][i], (prediction_box[0][i][0][0], prediction_box[0][i][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        return img
