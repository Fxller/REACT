import { useState } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import logo from '@/assets/react-logo.png';
import './App.css';

const API_URL = '/process';

export default function App() {
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Ciao! Come posso aiutarti oggi?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: userMessage.text })
      });
      const data = await response.json();
      let botText = 'Errore nella risposta';

      if ('rating' in data && !('review' in data)) {
        const labels = ['❌ Negativo', '😐 Neutro', '✅ Positivo'];
        const rating = parseInt(data.rating);
        const label = labels[rating] || 'Sconosciuto';
        botText = `La recensione è stata valutata come: ${label}`;
      } else if ('review' in data) {
        const reviewText = data.review.trim();

        const titleMatch = reviewText.match(/title:\s*(.+?)(?:\n|review:|$)/i);
        const reviewMatch = reviewText.match(/review:\s*(.+)/i);

        const title = titleMatch ? titleMatch[1].trim() : null;
        const reviewBody = reviewMatch ? reviewMatch[1].trim() : reviewText;

        botText = (
          <>
            <div className="review-title font-semibold mb-1">📝 <strong>{title || 'Recensione'}</strong></div>
            <div className="review-body">{reviewBody}</div>
          </>
        );
      }
      else {
        // Prova a estrarre prodotto e stelle per dare un messaggio più utile
        const matchProd = input.match(/per\s+"(.*?)"/i);
        const matchStar = input.match(/con\s+(\d)\s+stelle/i);

        const product = matchProd ? matchProd[1] : null;
        const stars = matchStar ? matchStar[1] : null;

        if (product && stars) {
          botText = `Sto generando una recensione per "${product}" con ${stars} stelle, ma qualcosa è andato storto.`;
        } else {
          botText = Object.values(data).join(' | ') || botText;
        }
      }

      const botMessage = {
        sender: 'bot',
        text: botText
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      setMessages(prev => [...prev, { sender: 'bot', text: `Errore: ${error.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app-container">
      <Card className="chat-card">
        <div className="chat-disclaimer">
          🛈 Le recensioni da classificare o i prodotti per cui vanno generate delle recensioni devono essere tra virgolette: <code>"..."</code>
        </div>
        <ScrollArea className="chat-scroll">
          <div className="chat-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`message ${msg.sender}`}>{msg.text}</div>
            ))}
            {loading && (
              <div className="message bot italic text-sm text-gray-400">Sto generando la risposta...</div>
            )}
          </div>
        </ScrollArea>
        <div className="chat-input-area">
          <div className="input-wrapper">
            <img src={logo} alt="React Logo" className="input-logo" />
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Scrivi un prompt..."
            />
          </div>
          <Button onClick={sendMessage} disabled={loading}>Invia</Button>
        </div>
      </Card>
    </div>
  );
}
