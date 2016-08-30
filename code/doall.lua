--------------------------------------------------------------------- 
--
-- This script performs training and testing on 
-- covnet model on a
-- classification problem. 
--

--
-- Ritambhara Singh
----------------------------------------------------------------------
require 'torch'

---------------------------------------------------------------------
print '==> processing options'

local set = function()
      cmd = torch.CmdLine()
      cmd:text()
      cmd:text('DeepChrome Pipeline options')
      cmd:text()
      cmd:text('Options:')

      -- global:
      cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
      cmd:option('-threads', 2, 'number of threads')
      cmd:option('--epochs', 100, 'number of epochs')
      --data:
      cmd:option('-dataDir', "data/", 'The data home location')
      cmd:option('-dataset', "toy/", 'Dataset name, corresponds to the folder name in dataDir')
      cmd:option('-resultsDir', "results/", 'The data home location')
      cmd:option('-name', "", 'Optionally, give a name for this model')
      cmd:option('-trsize', "10", 'Training set size (number of genes)')
      cmd:option('-tssize', "10", 'Test set size (number of genes)')

      --model:
      cmd:option('-nonlinearity', 'relu', 'type of nonlinearity function to use: tanh | relu | prelu')
      cmd:option('-loss', 'nll', 'type of loss function to minimize: nll')
      cmd:option('-model', 'convnet', 'type of model : linear|mlp|convnet')
      cmd:option('-nhus', '128', 'Number of hidden units in each layer')
      cmd:option('-ipdim', '1', 'Input dimension for spatial convolution')
      cmd:option('-opdim1', '6', 'Output dimension for spatial convolution')
      cmd:option('-opdim2', '16', 'Output dimension for spatial convolution')
      cmd:option('-FCnhus1', '120', 'Number of hidden units for fully connected layers')
      cmd:option('-FCnhus2', '84', 'Number of hidden units for fully connected layers')
      cmd:option('-conv_kernels', '5', 'Window size of mlp or convolution kernel sizes of conv')
      cmd:option('-pools', '2' , 'sizes of pooling windows')

      -- training:
      cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
      cmd:option('-plot', false, 'live plot')
      cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
      cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
      cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
      cmd:option('-momentum', 0, 'momentum (SGD only)')
      cmd:option('-type', 'double', 'type: double | float | cuda')
      cmd:text()
      opt = cmd:parse(arg or {})
end

set()

-- nb of threads and fixed seed (for repeatable experiments)

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)

----------------------------------------------------------------------
print '==> executing all'
dofile '1_data.lua'
dofile '2_model.lua'
dofile '3_loss.lua'
dofile '4_train.lua'
dofile '5_test.lua'

----------------------------------------------------------------------
print '==> training!'
local i = 1

while i<opt.epochs do
   train()
   test()
   i=i+1
end
