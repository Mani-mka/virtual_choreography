# import numpy as np
import csv
import torch
from torch.utils import data
from torch import nn
import torchvision.transforms as transforms

path = "D:/Projects/Major project/csv_keypoints/KeypointData.csv"

no_columns = 41
no_meta_data_rows = 2
no_meta_data_columns = 5
limit = 20000
train_data = []


def csv_read(path1):
    with open(path1) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if csv_reader.line_num in range(no_meta_data_rows):
                pass
            else:
                temp1 = []
                for i in range(no_meta_data_columns, no_columns, 2):
                    temp2 = []
                    for j in range(i, i+2):
                        temp2.append(float(row[j]))
                    temp1.append(temp2)
                train_data.append(temp1)
            if csv_reader.line_num > limit:
                break


csv_read(path)
# print(np.shape(train_data))
# plt.scatter(train_data[12000][:][0], train_data[12000][:][1])
# print(train_data[12000][2:6])
# plt.show()

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_labels = torch.zeros(limit, dtype=torch.double)

train_data = torch.tensor(train_data, dtype=torch.float64)

train_set = data.TensorDataset(train_data, train_labels)


# class CustomKeyPointDataset(data.Dataset):
#     def __init__(self, old_train_set, new_transform=None):
#         self.train_set = old_train_set
#         self.transform = new_transform
#         # self.train_set = self.transform(old_train_set)
#
#     def __len__(self):
#         return len(self.train_set)
#
#     def __getitem__(self, idx):
#         keypoint = self.transform(self.train_set[idx][0])
#         sample = (keypoint, self.train_set[idx][1])
#         return sample


batch_size = 32

# train_set_new = CustomKeyPointDataset(old_train_set=train_set, new_transform=transform)

train_loader = data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)

# print(train_loader.dataset.__getitem__(2))


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(36, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), 36)
        output = self.model(x)
        return output


discriminator = Discriminator()


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 36),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 18, 2)
        output.type(dtype=torch.FloatTensor)
        return output


generator = Generator()

lr = 0.001
num_epochs = 50
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples = real_samples.type(dtype=torch.FloatTensor)
        real_samples_labels = torch.ones((batch_size, 1))
        latent_space_samples = torch.randn((batch_size, 100))
        generated_samples = generator(latent_space_samples)
        # print(real_samples[1])
        generated_samples_labels = torch.zeros((batch_size, 1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 100))

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        if epoch % 2 == 0 and n == batch_size - 1:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
