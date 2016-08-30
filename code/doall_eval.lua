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

      cmd:option('-epoch', 1, 'epoch number of best model')
      --data:
      cmd:option('-dataDir', "data/", 'The data home location')
      cmd:option('-dataset', "toy/", 'Dataset name, corresponds to the folder name in dataDir')
      cmd:option('-resultsDir', "results/", 'The data home location')
      cmd:option('-name', "", 'Optionally, give a name for this model')
      cmd:option('-tssize', "9", 'Test set size (number of genes)')

      opt = cmd:parse(arg or {})
end

set()


----------------------------------------------------------------------
print '==> executing all'
dofile '1_data.lua'
--perform testing:
dofile '6_eval.lua'
--perform visualization
dofile '7_viz.lua'

----------------------------------------------------------------------
print '==> evaluation!'

test()

print '==> visualization!'
viz_model() 
