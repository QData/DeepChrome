----------------------------------------------------------------------
--
-- This script performs training and testing for evaluation on 
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

      cmd:option('-epoch', 5, 'epoch number of best model')
      --data:
      cmd:option('-dataDir', "/bigtemp/rs3zz/DeepChrome/", 'The data home location')
      cmd:option('-dataset', "toy/classification", 'Dataset name, corresponds to the folder name in dataDir')
      cmd:option('-resultsDir', "/bigtemp/rs3zz/DeepChrome/results/", 'The data home location')
      cmd:option('-name', "", 'Optionally, give a name for this model')
      cmd:option('-tssize', "9", 'Test set size (number of genes)')

      opt = cmd:parse(arg or {})
end

set()

-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end

----------------------------------------------------------------------
print '==> executing all'
dofile '1_data.lua'
dofile '8_featmap.lua'

----------------------------------------------------------------------
print '==> evaluation!'

test()
 
