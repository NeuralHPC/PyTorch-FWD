# Hyperparameters
### CelebAHQ - 64x64
| Hyper parameter |    Values |
|-----------------|-----------|
| Epochs          |    280    |
| Seed            |     42    |
| resize          |     64    |
| channel-mult    | '1,2,2,4' |
| num-res-blocks  |     2     |
| attn-heads      |     4     |
| attn-resolution |   '16,8'  |
| base-channels   |    128    |
| time-steps      |    1000   |
| batch-size      |     16    |


### CelebA - 64x64
| Hyper parameter | Values      |
|-----------------|-------------|
| Epochs          | 300         |
| Seed            | 42          |
| resize          | 64          |
| channel-mult    | '1,2,2,2,4' |
| num-res-blocks  | 2           |
| attn-heads      | 4           |
| attn-resolution | '16,'       |
| base-channels   | 128         |
| time-steps      | 1000        |
| batch-size      | 128         |

### CelebAHQ - 128x128
| Hyper parameter |    Values |
|-----------------|-----------|
| Epochs          |    280    |
| Seed            |     42    |
| resize          |     128   |
| channel-mult    |'1,2,2,4,4'|
| num-res-blocks  |     3     |
| attn-heads      |     4     |
| attn-resolution |   '16,8'  |
| base-channels   |    128    |
| time-steps      |    1000   |
| batch-size      |     16    |
