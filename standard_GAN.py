
# standard-GAN

# -----------------------------------------------------------------------------------------
# This code forms part of the graded assignment from the "Deep Learning II: Master Data Science"
# course, completed by a member of our team.
#
# Primary Contributions:
# - The core implementation, which was subject to grading, was developed by the student.
#   Although the overall code structure was inspired by the course instructor's template.
# - Personal enhancements include:
#   - Correction and improvement of the original coursework.
#   - Adaptation from Keras to PyTorch to maintain consistency across our project's codebase.
#
# Code Enhancement:
# - This code has been enhanced to align with the Defense-GAN (WGAN) approach, approache which is built 
#   using a standard GAN.
#   as introduced in the paper "DEFENSE-GAN: PROTECTING CLASSIFIERS AGAINST ADVERSARIAL ATTACKS USING
#   GENERATIVE MODELS" by Pouya Samangouei, Maya Kabkab, and Rama Chellappa (2018).
# -----------------------------------------------------------------------------------------


# Define the Generator network
class Generator(nn.Module):
    def __init__(self, z_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        flattened = img.view(img.size(0), -1)
        validity = self.model(flattened)
        return validity

class GAN():
    def __init__(self, dataset_name='mnist'):
        # Load data
        self.img_shape = (1, 64, 64)  # for MNIST
        self.z_dim = 100
        self.dataset_name = dataset_name
        self.model_file = f'models/{self.dataset_name}_gan_model.pickle'

        # Define networks
        self.generator = Generator(self.z_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)

        # Optimizers
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Loss function
        self.adversarial_loss = nn.BCELoss()

    def load_gan_data(self):
        # MNIST Dataset
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        return dataloader

    def train(self, epochs, train_loader, sample_interval=1000):
        for epoch in range(epochs):
            for i, (imgs, _) in enumerate(train_loader):
                # Adversarial ground truths
                valid = Variable(torch.FloatTensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(torch.FloatTensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(torch.FloatTensor))

                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.size(0), self.z_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

                # If at save interval => save generated image samples and model checkpoints
                if i % sample_interval == 0:
                    # Save image samples
                    # Save model checkpoints                    
                    self.save_sample_images(epoch, i)
                    torch.save(self.generator.state_dict(), f'results/generator_epoch{epoch}_batch{i}.pth')
                    torch.save(self.discriminator.state_dict(), f'results/discriminator_epoch{epoch}_batch{i}.pth')                
                
    def save_sample_images(self, epoch, batch):
        # Generate noise
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (25, self.z_dim))))

        # Generate images from noise
        gen_imgs = self.generator(z).detach()

        # Rescale images from [-1, 1] to [0, 1] range
        gen_imgs = (gen_imgs + 1) / 2

        # Save image grid
        save_image(gen_imgs.data, f'results/epoch{epoch}_batch{batch}.png', nrow=5, normalize=True)
