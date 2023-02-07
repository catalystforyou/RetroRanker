import time
from datetime import datetime

import torch
import torch.nn as nn

from retro_dataloader import build_dataloader
from retro_ranker import RetroRanker
from utils.common_utils import parse_training_config
from utils.model_utils import build_ckpt_path, load_model, move_to_device


def predict(model, bg, bg_golden, bg_products, bg_product_golden, masks, device=None):
    # mask_node, mask_edge
    mol1 = [bg_golden, bg_product_golden, masks[2], masks[3], masks[4][1]]
    mol2 = [bg, bg_products, masks[0], masks[1], masks[4][0]]
    if device:
        mol1 = move_to_device(mol1, device)
        mol2 = move_to_device(mol2, device)
    return model(mol1, mol2)


def predict_score(model, bg, bg_golden, bg_products, bg_product_golden, masks, device=None):
    mol2 = [bg, bg_products, masks[0], masks[1], masks[4][0]]
    if device:
        mol2 = move_to_device(mol2, device)
    return model(mol2)


def run_a_train_epoch(args, epoch, model, data_loader, optimizer):
    model.train()
    train_loss = 0
    train_acc = 0
    starttime = time.time()
    for batch_id, batch_data in enumerate(data_loader):
        bg, bg_golden, bg_products, bg_product_golden, masks = batch_data
        loss, output = predict(model, bg, bg_golden,
                               bg_products, bg_product_golden, masks, args.device)
        output = output.detach().cpu().numpy()
        train_acc += len(output[output[:, 0] < output[:, 1]])/len(output)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_clip)
        optimizer.step()

        if batch_id % args.print_every == 0:
            print('\r %s epoch %d/%d, batch %d/%d, loss %.4f, acc %.4f' % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch + 1, args.num_epochs, batch_id + 1, len(
                data_loader) // args.batch_size + 1, train_loss/(batch_id+1), len(output[output[:, 0] < output[:, 1]])/len(output)), end='', flush=True)
    print('\n'+'Epoch time:', time.time() - starttime)
    starttime = time.time()
    print('\nepoch %d/%d, training loss: %.4f, training acc: %.4f' % (epoch +
          1, args.num_epochs, train_loss/(batch_id+1), train_acc/(batch_id+1)))


def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            bg, bg_golden, bg_products, bg_product_golden, masks = batch_data
            loss, output = predict(
                model, bg, bg_golden, bg_products, bg_product_golden, masks, args.device)
            output = output.detach().cpu().numpy()
            val_acc += len(output[output[:, 0] < output[:, 1]])/len(output)
            val_loss += loss.item()
            
    return val_loss/(batch_id+1), val_acc/(batch_id+1)





def main(args):
    model = RetroRanker(node_out_feats=args.out_node_feats)
    model, optimizer, scheduler, stopper = load_model(model, args, True)
    train_loader, val_loader = build_dataloader(args, True, args.dataset)
    for epoch in range(args.num_epochs):
        run_a_train_epoch(args, epoch, model, train_loader, optimizer)
        val_loss, val_acc = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_acc, model)
        scheduler.step()
        print('epoch %d/%d, validation loss: %.4f, validation acc: %.4f' %
              (epoch + 1, args.num_epochs, val_loss, val_acc))
        print('epoch %d/%d, Best loss: %.4f' %
              (epoch + 1, args.num_epochs, stopper.best_score))
        torch.save(model.state_dict(), build_ckpt_path(args, epoch))
        if early_stop:
            print('Early stopped!!')
            break


if __name__ == '__main__':
    args = parse_training_config()
    args.device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(args)
    main(args)
