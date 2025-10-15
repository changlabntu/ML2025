#First, download medical_mnist.zip
#from Kaggle https://www.kaggle.com/datasets/andrewmvd/medical-mnist or
#from dropbox https://www.dropbox.com/scl/fi/wql10vir4zk5mppc88nl8/medical_mnist.zip?rlkey=majqo92r2g5uejqmsnmlo0l5j&dl=0
#unzip the data to Exercise2 / mmnist/
#import all the helper functions

from utils_mnist import *
import numpy as np


class Trainer_template:
    def __init__(self):
        """ a __init__ method that every python class need."""
        pass

    def overall_loop(self):
        """Run training for specified epochs, print train/val loss each epoch."""
        pass

    def training_loop(self, train_loader):
        """Iterate through train batches, compute losses. Returns list of losses."""
        pass

    def validation_loop(self, validation_loader):
        """Iterate through validation batches, compute losses. Returns list of losses."""
        pass

    def training_step(self, train_batch):
        """Forward pass, compute loss, backprop, update weights. Returns loss."""
        pass

    def validation_step(self, validation_batch):
        """Forward pass, compute loss (no backprop). Returns loss."""
        pass


class Trainer():
    def __init__(self): #, args, train_loader, validation_loader, model, loss_function, optimizer):
        pass

    def overall_loop(self):
        for epoch in range(args['num_epochs']):
            train_loss = self.training_loop(train_loader)  # do the training look
            validation_loss = self.validation_loop(validation_loader) # do the validation loop
            # print out the training and validation loss per epoch
            print('Epoch [{}/{}], Train Loss: {:.4f}, Validation Loss: {:.4f}'
                  .format(epoch + 1, args['num_epochs'],
                          sum(train_loss) / len(train_loss), sum(validation_loss) / len(validation_loss)))

    def training_loop(self, train_loader):
        train_loss = []
        for i, train_batch in enumerate(train_loader):  # get mini-bacthes from the train_loader
            loss = self.training_step(train_batch) # get the train loss from the train_step
        train_loss.append(loss) # add the train_loss up
        return train_loss

    def validation_loop(self, validation_loader):
        validation_loss = []
        for i, validation_batch in enumerate(validation_loader):
            loss = self.validation_step(validation_batch)
        validation_loss.append(loss)
        return validation_loss

    def training_step(self, train_batch):
        (images, labels) = train_batch
        images = images.reshape(-1, args['img_size'])  # turn the 2D image to a 1D feature (flatten)

        # Forward pass
        outputs = model(images)  # get the model output
        loss = loss_function(outputs, labels) # calculate the loss

        # Magic section where we do optimization by backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def validation_step(self, validation_batch):
        (images, labels) = validation_batch
        images = images.reshape(-1, args['img_size'])

        # Forward pass
        outputs = model(images)  # get the model output
        loss = loss_function(outputs, labels)  # calculate the loss
        return loss


# arguments
def get_arguments():
    # Hyper-parameters
    args = {'img_size': 64 * 64,
            'num_classes': 10,
            'num_epochs': 50,
            'batch_size': 16,
            'learning_rate': 0.001,
            'model': 'logistic_regression'} # MLP or logistic_regression
    return args


args = get_arguments()
print(args)

# Medical MNIST dataset (images and labels)
train_loader, validation_loader = get_medical_mnist(args=args)
print('Done with data preparation')

print('Length of train dataset:')
print(len(train_loader.dataset))
print('Length of validation dataset:')
print(len(validation_loader.dataset))
print('Length of train dataloader:')
print(len(train_loader))
print('Length of validation dataloader:')
print(len(validation_loader))

some_index = np.random.randint(0, len(train_loader.dataset), 10)
some_imgs = [train_loader.dataset.__getitem__(idx)[0] for idx in some_index]
some_labels = [train_loader.dataset.__getitem__(idx)[1] for idx in some_index]

show_examples(some_imgs[:5], some_labels[:5])

# get model
if args['model'] == 'logistic_regression':
    print('Using logistic regression')
    model = nn.Linear(args['img_size'], args['num_classes'])
elif args['model'] == 'MLP':
    print('MLP')
    model = MLP(dropout=0, hidden_1=512, hidden_2=512)

# Loss and optimizer
loss_function = nn.CrossEntropyLoss() # this combined the LogSoftmax and NLLLoss
optimizer = torch.optim.SGD(model.parameters(), lr=args['learning_rate'])

trainer = Trainer()#args, train_loader, validation_loader, model, loss_function, optimizer)

trainer.overall_loop()

correct = 0
total = 0
for images, labels in validation_loader:
    images = images.reshape(-1, args['img_size'])
    outputs = model(images)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print(correct)
print(total)

print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))