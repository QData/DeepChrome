----------------------------------------------------------------------
-- This script defines following loss function : 
--   
--   + negative-log likelihood, using log-normalized output units (SoftMax)

--
-- Ritambhara Singh
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of loss functions

----------------------------------------------------------------------

-- classification problem

noutputs = 2

----------------------------------------------------------------------
print '==> define loss'

model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
print '==> here is the loss function:'
print(criterion)

--------------------------- End of Code -------------------------------
