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

- why isnt enumerate running in parallel?

- Arthur's test case: 

- start 


## later
- ditch triple eq
- add in basic math for constructing numbers and stuff
- expand vocab but dont have prim chars...must be referenced to be used

- add an explicit reflection of the state timestep so that get_historical 
  can be used to reference a fixed index in constant bits



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