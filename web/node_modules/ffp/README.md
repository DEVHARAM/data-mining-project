# Farthest feasible point

> FFP algorithm implementation.

Algorithm is used to filter timeseries to reduce count of points by defining maximum error on ordinate axis.

<img src="https://raw.githubusercontent.com/devgru/ffp/master/demo.png" alt="FFP Demo" width="400">

FFP algorithm is described in [this](http://masc.cs.gmu.edu/wiki/uploads/GuilinLiu/ffp.pdf) paper, authored by Guilin Xinyu and Zhe Cheng.

Library is distributed as UMD module.

Check [spec/ffp.spec.js](spec/ffp.spec.js) to see usage example.

Built in collaboration with [Erohina Elena](https://github.com/erohinaelena), original version of FFP implementation can be found [here](http://bl.ocks.org/erohinaelena/882e7cadc2fd687cf2b3).

## Installing

```sh
$ yarn add ffp
# or
$ npm install --save ffp
```

## Usage

FFP is used like this:

```js
import FFP from 'ffp';
const ffp = FFP()
  .maxDelta(0.5)
  .x(({x}) => x)
  .y(({y}) => y)
  .result(({item}) => item);
```

FFP library exports a function.

## FFP()

Creates FFP utility.

## ffp(array)

FFP utility is a function, invoke it on array of elements to filter them out.

## ffp.maxDelta([*delta*])

If *delta* is specified, sets the maximum *delta* to the specified number. If *delta* is not specified, returns the current maximum *delta* value, which defaults to 1.

Maximum *delta* defines maximum error between resulting trend and point position on ordinate axis.

## ffp.epsilon([*epsilon*])

If *epsilon* is specified, sets the *epsilon* to the specified number. If *epsilon* is not specified, returns the current *epsilon* value, which defaults to 1/2³².

*epsilon* defines maximum calculation error.

## ffp.x([*x*])

If *x* is specified, sets the *x* accessor to the specified function. If *x* is not specified, returns the current *x* accessor, which defaults to `(value, index) => index`.

*x* accessor is invoked for each point.

## ffp.y([*y*])

If *y* is specified, sets the *y* accessor to the specified function. If *y* is not specified, returns the current *y* accessor, which defaults to `(value) => value`.

*x* accessor is invoked for each point.

## ffp.result([*result*])

If *result* is specified, sets the *result* accessor to the specified function. If *result* is not specified, returns the current *result* accessor, which defaults to `(value) => value`.

*result* accessor is invoked for each point. By default, result of `ffp(array)` call is an array of objects with `item` and `index` keys. Define *result* accessor to modify this behavior.

## Development

* Run tests: `yarn test`;
* Build `yarn build`;

## License

MIT © [Dmitriy Semyushkin](https://devg.ru)
