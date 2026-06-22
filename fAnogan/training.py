import os
import torch
import torch.nn as nn
import torch.autograd as autograd
from tqdm import tqdm


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.rand(*real_samples.shape[:1], 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    d_interpolates = D(interpolates)
    fake = torch.ones(*d_interpolates.shape, device=device)
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_wgangp(args, generator, discriminator, dataloader, device, lambda_gp=10):
    os.makedirs(os.path.join(args.ad_save_path, args.env), exist_ok=True)

    generator.to(device)
    discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_ad, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_ad, betas=(0.5, 0.999))

    padding_epoch = len(str(args.epochs_ad))
    padding_i = len(str(len(dataloader)))

    for epoch in range(args.epochs_ad):
        pbar = tqdm(enumerate(dataloader))
        for i, x in pbar:
            real_data = x.to(device)

            optimizer_D.zero_grad()
            z = torch.randn(x.shape[0], args.fanogan_latent_dim, device=device)
            fake_data = generator(z)

            real_validity = discriminator(real_data)
            fake_validity = discriminator(fake_data.detach())
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_data.data, fake_data.data, device)
            d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity)
                      + lambda_gp * gradient_penalty)

            d_loss.backward()
            optimizer_D.step()
            optimizer_G.zero_grad()

            if i % args.fanogan_n_critic == 0:
                fake_data = generator(z)
                fake_validity = discriminator(fake_data)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                pbar.set_description(
                    f"[Epoch {epoch:{padding_epoch}}/{args.epochs_ad}] "
                    f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():3f}] "
                    f"[G loss: {g_loss.item():3f}]"
                )

    torch.save({
        'Generator': generator.state_dict(),
        'Discriminator': discriminator.state_dict(),
    }, os.path.join(args.ad_save_path, args.env, 'gan.pth'))


def train_encoder_izif(args, generator, discriminator, encoder, dataloader, device, kappa=1.0):
    os.makedirs(os.path.join(args.ad_save_path, args.env), exist_ok=True)

    statedict = torch.load(os.path.join(args.ad_save_path, args.env, 'gan.pth'))
    generator.load_state_dict(statedict['Generator'])
    discriminator.load_state_dict(statedict['Discriminator'])

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device)

    criterion = nn.MSELoss()
    optimizer_E = torch.optim.Adam(encoder.parameters(), lr=args.lr_ad, betas=(0.5, 0.999))

    padding_epoch = len(str(args.epochs_ad))
    padding_i = len(str(len(dataloader)))

    for epoch in range(args.epochs_ad):
        pbar = tqdm(enumerate(dataloader))
        for i, x in pbar:
            real_data = x.to(device)

            optimizer_E.zero_grad()
            z = encoder(real_data)
            fake_data = generator(z)

            real_features = discriminator.forward_features(real_data)
            fake_features = discriminator.forward_features(fake_data)

            loss_data = criterion(fake_data, real_data)
            loss_features = criterion(fake_features, real_features)
            e_loss = loss_data + kappa * loss_features

            e_loss.backward()
            optimizer_E.step()

            if i % args.fanogan_n_critic == 0:
                pbar.set_description(
                    f"[Epoch {epoch:{padding_epoch}}/{args.epochs_ad}] "
                    f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                    f"[E loss: {e_loss.item():3f}]"
                )

    torch.save(encoder.state_dict(), os.path.join(args.ad_save_path, args.env, 'encoder.pth'))
