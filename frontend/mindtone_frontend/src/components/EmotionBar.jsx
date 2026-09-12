export const EMOTION_COLORS = {
  Angry: '#E74C3C',
  Disgust: '#8E44AD',
  Fear: '#F39C12',
  Happy: '#2ECC71',
  Neutral: '#95A5A6',
  Sad: '#3498DB',
};

export const EMOTION_EMOJI = {
  Angry: '😠',
  Disgust: '🤢',
  Fear: '😨',
  Happy: '😊',
  Neutral: '😐',
  Sad: '😢',
};

export default function EmotionBar({ emotion, probability }) {
  const color = EMOTION_COLORS[emotion] || '#999';
  return (
    <div className="mb-2">
      <div className="mb-1 flex items-center justify-between text-sm">
        <span className="font-medium text-teal-900">
          {EMOTION_EMOJI[emotion]} {emotion}
        </span>
        <span className="text-teal-900/60">{(probability * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-teal-50">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${probability * 100}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}
