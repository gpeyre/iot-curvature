import {sampleData, solveAt, curve} from './math.mjs';
let activeData;
self.onmessage = ({data: message}) => {
  try {
    if (message.type === 'sample') {
      activeData = sampleData(message.config);
      const other = message.compare ? sampleData({...message.config,
        sampling: message.config.sampling === 'paired' ? 'independent' : 'paired'}) : null;
      self.postMessage({type: 'points', id: message.id, data: activeData,
        current: solveAt(activeData, message.u)});
      const values = curve(activeData, 41, (done, total) => {
        if (done % 5 === 0) self.postMessage({type: 'progress', id: message.id,
          value: done / (total * (other ? 2 : 1))});
      });
      const otherValues = other ? curve(other, 41, (done, total) => {
        if (done % 5 === 0) self.postMessage({type: 'progress', id: message.id,
          value: 0.5 + done / (2 * total)});
      }) : null;
      self.postMessage({type: 'curves', id: message.id, values, otherValues,
        otherSampling: other?.config.sampling, current: solveAt(activeData, message.u)});
    } else if (message.type === 'current' && activeData) {
      self.postMessage({type: 'current', id: message.id, current: solveAt(activeData, message.u)});
    }
  } catch (error) { self.postMessage({type: 'error', id: message.id, error: error.message}); }
};
