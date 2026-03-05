## NEXT
x make get hist obs relative to the current timestep by default!!

x get hist could look at the future!!!
  x need to pass versioned versions of state instead of this global lookup stuff
     x finish reviewing changes
        x wtf does evaluate() do?? compile? how so?: 
            okay - it lets environmental conditions be used to change what 
            function is actually called via isConditional logic...but why?
            mainly this is used to assign the variables to Index nodes so 
            that a callable can be constructed

- review new test results...some weird failures??
  x make a nice plot of which programs enter the pool and when (and frozen 
    store)
  x how is the pool used for prediction? - just takes the first program for now
- runs of 3 test: should have recovered some predictors in the second half...
  x is this because of task duplicate filtering??
  x increase budget and timeout -> doesnt help
  x debug: dump enum when we fail to find solution
  x may need to add a few more primitives
    x get absolute hist obs
    x get timestep
    x arithmetic
  x auto-run graph gen 
  x log runs to separate folders with timestamps
  x cleaned up enum debug 


x why isn't enumerate running in parallel?
  x ported to bottom-up solver
x restore enum debug hooks

- auto clean up enum logs
- store git hash and diffs with experiment
  - just use aim to record and vis stuff?
   

- conditional eval...does it actually make sense? not see any ifs in the 
  enum, might be that we can compile it out 
  - make an "if" that isnt detected as conditional so we do eval both paths
    at runtime?
  - BETTER: print out programs in pre-compiled state
  - get this working before trying MNIST which might require functioning 
    branches

  - need loss ranking in pool to deal with random (unpredictable) sequences

- Arthur's test case: 
  - make a longer random string
  - add prims that could output characters not just succ-char




    
## paper plots
- test case illustrations
  - in order to show recovery times, we'll have to mesh the enumeration 
    schedule differently
    - ideas: 

## later
- ditch triple eq
- add in basic math for constructing numbers and stuff
- expand vocab but dont have prim chars...must be referenced to be used

- add an explicit reflection of the state timestep so that get_historical 
  can be used to reference a fixed index in constant bits

- add frozen program store to the timeline plot


## ideas

- if we always exclude old programs that got disqualified, does that force 
  us to eventually invent new programs that cover them more generally vs 
  re-surfacing them?
  - not really...if its 'load the k-th previous program and do something 
    with it' and this has been tried before, but the memory state has 
    changed so now it has a different meaning....so we should keep it! 
  - is this tradeoff worth it?
  - [ ] TODO: disable


- visualization of amortization
  - sample the program space and plot the mass of the recognition network
    - plot as graph...each node is either sized based on prob mass or shaded 
      that way
      - draw lines for popular samples
      - then make movies of this evolution!


## MNIST path
x run it
- make model more realistically sized
- look at program store
    - add program store to timeline view
- timeline view:
  - debug log loss display...what is it, actually?
  - can I embed visualizations of the output distribution?
- make a movie of pool contents, obs, preds, perf, etc
- why does uniform pred win?
- port to bottom solver

- cache backpropped params somehow
- ensure cheap to update most recent model
- ensure cheap to find and update the "old" model
- 


- exciting experiment:
  - equip neural path with knn lookups and a broader set of options for 
    selection compute grap


- incorp recognition and dream phases?