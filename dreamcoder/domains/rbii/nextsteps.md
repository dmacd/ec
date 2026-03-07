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


## full RBII loop

- baseline_bits == val_window? NO...its the validation
  x > assumes each observation is worth one bit, which isnt true
    > but does it matter? with fixed finite alphabet and boolean 
    > comparisons
  x TODO: refactor with distribution-typed outputs
  x weird: cand.witness bits should NEVER be none!! thats an exception
  x using hard min_weight cutoff - not what I want
  x candidate buffer cap...wtf? then i have to align this with the proposal 
    batch size...which...ugh. no. fuck this
  x I also want to support a mode where we just search for stuff...and if it 
    is more worthy than what's in the pool, we add it
  
  - shouldn't the window loss have been computed by the solver for the task!?
  - why is rerank so overwrought?? i should jush have a list of programs and 
    scores....sort and take...really???
  x dont add duplicates to the pool if they are already there
  x wtf? why are we adding to best programs every candidate???
  x why are we normalizing pool weights????? is that in teh manuescript?
  x frozen programs are what should be store in the state, not the RBIILoop!
    - candidates should not get recorded in best/frozen programs automatically
    
  x is the candidate buffer getting rolled over??
    x no, we're just looking at the last -n:0 after extending it on each search
    x this blows, lets nuke it and start over each cycle
  - MDL filter missing? 
    x need explicit computation of compression gain 
    x slack param
    x- need to generalize to non-binary observations
       x ugh, this is way too much code wtf 
       x finish reviewing loss
           x method for inferring alphabet is super dumb
  - then run it?

x viz fixes
  x need frozen id and pool id separate!
  x separate elbows


- "force_if" sequence
  - why does it fail totally? 
    - what is keeping any programs from being accepted?
      x maybe the MDL criterion is too strict when a "good" solution isn't 
        quite possible?
      - maybe the program that compares the last 3 symbols to a and predicts 
        b is not accesible?
        - we cant construct characters directly so it will already be very hard!
    - > what if programs are allowed to express "no opinion" ?? 
  - why does it get some partial bits extracted but not continue and notice 
    the actual pattern?
    - probably need better incremental transforms i.e. real transformers
    - logical NOT would probably help too?
    - double check if conds are being evaluated properly at runtime...they 
      might NOT be using the right state????
  - try: make it easier "aaaaaabdddaaaaabdaaaaabdddd...." (random number of 
    d's to prevent cycle dectection but not so many its hard to pick up)
  - **double-check that ifs are actually evaluated with the right arguments!**
    - try eager-if variant that doesnt get compiled

- experimental questions
  - will increasing the pool size give borderline programs more 
    opportunity to become proven i.e. edge out competitors because they have 
    more tolerance to noise or small differences in performance?

## streaming viz 
x make a javascript explorer that lets us probe and unfold stuff?
x key feature --> streaming update?
x MUST stream events from server, cant send the whole thing at once
x must NOT render svg -- use html / js elements for layout, must be dynamic



## transformers

- edit op that takes a path sequence and then substitutes the function or 
  primitive at that path, if the types line up
- do we freeze the mutations as prims and not just the predictors???
  - we want to reuse the info in the mutations as well, so why not?
- 
- we probably want to use pool predictors for rec and dreaming, prims 
  library, not just the frozen ones!

- plan notes
  - drop the old get_program, should be eval by ref then apply
  - edit_replace_with_hist_obs should not just be any obs type, same with 
    programs
    - i.e. dont duplicate prims for edits, let the edit ref the path to it 
      then replace with compatible type
    - a wrap edit makes sense though
  - Abstraction??

- ugh....complex...requires quoting, effectively, and thats annoying
- what about a separate transformer pass where we enumerate edits by making 
  holes in the existing programs?
  - next step: actually review the current search methodology in more detail
    - ask gpt pro to come up with a strategy that is simple

- how do i fold prims into the library?
  - 

## test cases

- have runs of increasing repeat, measure returning perf
- construct a CL-ish test case with random substrings that repeat in random 
  orders and measure their error and return time



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

