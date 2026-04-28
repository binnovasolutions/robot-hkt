import asyncio
import json
import os
import base64
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv
import websockets
from openai import AsyncOpenAI

# Load biến môi trường từ file .env
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# ==========================================
# ⚙️ CẤU HÌNH ELEVENLABS
# ==========================================
ELEVENLABS_MODEL = "eleven_flash_v2_5" 
ELEVENLABS_STABILITY = 0.35      
ELEVENLABS_SIMILARITY = 0.85     
ELEVENLABS_STYLE = 0.4           
ELEVENLABS_SPEAKER_BOOST = True  

app = FastAPI()
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

def get_system_prompt():
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        default_prompt = "Bạn là một Robot AI thân thiện, thông minh. Chỉ trả lời kết quả là text, không chèn ký tự đặc biệt. Có dấu phẩy, chấm phẩy, ngắt nghỉ mỗi câu nói."
        with open("prompt.txt", "w", encoding="utf-8") as f:
            f.write(default_prompt)
        return default_prompt

async def keep_deepgram_alive(dg_socket, client_websocket, session_state):
    """Luồng chạy ngầm: Quản lý Timeout 20s và chủ động ngắt kết nối"""
    try:
        keepalive_count = 0
        bot_working_keepalive_timer = 0
        while True:
            await asyncio.sleep(1) 
            
            if session_state.get("is_bot_working", False):
                bot_working_keepalive_timer += 1
                if bot_working_keepalive_timer >= 5:
                    await dg_socket.send(json.dumps({"type": "KeepAlive"}))
                    bot_working_keepalive_timer = 0
                session_state["last_audio_time"] = time.time()
                keepalive_count = 0
                continue

            bot_working_keepalive_timer = 0
            idle_time = time.time() - session_state.get("last_audio_time", time.time())
            
            if idle_time >= 20:
                print(f"\n[{time.strftime('%H:%M:%S')}] ⏱️ [TIMEOUT] Quá 20s không có tiếng nói. Đóng phiên!")
                try:
                    await client_websocket.send_json({"type": "control", "event": "deepgram_timeout"})
                    await dg_socket.close()
                    await asyncio.sleep(0.5) 
                    await client_websocket.close()
                except Exception:
                    pass
                break 

            elif idle_time >= 10 and keepalive_count == 1:
                await dg_socket.send(json.dumps({"type": "KeepAlive"}))
                keepalive_count = 2
            elif idle_time >= 5 and keepalive_count == 0:
                await dg_socket.send(json.dumps({"type": "KeepAlive"}))
                keepalive_count = 1
            elif idle_time < 5:
                keepalive_count = 0
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Lỗi KeepAlive: {e}")

async def deepgram_receiver(dg_socket, client_websocket, session_state):
    """Luồng chạy ngầm nhận text từ Deepgram và gửi về Web (ĐÃ VÁ LỖI)"""
    print(f"[{time.strftime('%H:%M:%S')}] 👂 [DEEPGRAM] Bắt đầu lắng nghe kết quả từ Deepgram...")
    try:
        async for message in dg_socket:
            try: # Bọc try-except bên trong vòng lặp để bảo vệ luồng không bị sập
                res = json.loads(message)
                
                # Kiểm tra chặt chẽ: gói tin phải là dictionary và có chứa key 'channel' dạng dictionary
                if isinstance(res, dict) and "channel" in res and isinstance(res["channel"], dict):
                    channel = res["channel"]
                    alternatives = channel.get("alternatives", [])
                    
                    if alternatives and isinstance(alternatives, list) and len(alternatives) > 0:
                        sentence = alternatives[0].get("transcript", "")
                        is_final = res.get("is_final", False)
                        
                        if sentence:
                            session_state["last_audio_time"] = time.time()
                            
                            if is_final:
                                print(f"\n[{time.strftime('%H:%M:%S')}] ✅ [DEEPGRAM-CHỐT]: '{sentence}'")
                                session_state["final_transcript"] += sentence + " "
                                session_state["interim_transcript"] = "" 
                            else:
                                print(f"\r[{time.strftime('%H:%M:%S')}] ⏳ [DEEPGRAM-TẠM]: '{sentence}'", end="", flush=True)
                                session_state["interim_transcript"] = sentence

                            await client_websocket.send_text(json.dumps({
                                "type": "transcript",
                                "text": sentence,
                                "is_final": is_final
                            }))
            except Exception as parse_err:
                # Bỏ qua các gói tin VAD Control Message lạ
                continue 

    except Exception as e:
        print(f"\n[{time.strftime('%H:%M:%S')}] 🔴 [DEEPGRAM-LỖI NGHIÊM TRỌNG] Luồng nhận bị đóng: {e}")

async def elevenlabs_tts_worker(tts_queue, client_websocket):
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream-input?model_id={ELEVENLABS_MODEL}&output_format=pcm_16000"
    print(f"[{time.strftime('%H:%M:%S')}] 🗣️ [ELEVENLABS] Đang kết nối TTS WebSocket...")
    try:
        async with websockets.connect(uri) as el_socket:
            print(f"[{time.strftime('%H:%M:%S')}] 🗣️ [ELEVENLABS] Đã kết nối thành công. Đang chờ text chunk...")
            init_msg = {
                "text": " ",
                "voice_settings": {
                    "stability": ELEVENLABS_STABILITY, 
                    "similarity_boost": ELEVENLABS_SIMILARITY,
                    "style": ELEVENLABS_STYLE,
                    "use_speaker_boost": ELEVENLABS_SPEAKER_BOOST
                },
                "xi_api_key": ELEVENLABS_API_KEY
            }
            await el_socket.send(json.dumps(init_msg))

            async def sender():
                while True:
                    text_chunk = await tts_queue.get()
                    if text_chunk is None:
                        print(f"\n[{time.strftime('%H:%M:%S')}] 🗣️ [ELEVENLABS] Đã nhận tín hiệu KẾT THÚC câu.")
                        await el_socket.send(json.dumps({"text": ""}))
                        break
                    if text_chunk.strip():
                        print(f"[{time.strftime('%H:%M:%S')}] 🗣️ [ELEVENLABS] Đang chuyển thành giọng nói đoạn: '{text_chunk}'")
                        await el_socket.send(json.dumps({
                            "text": text_chunk, 
                            "try_trigger_generation": True
                        }))

            async def receiver():
                audio_chunk_count = 0
                async for message in el_socket:
                    data = json.loads(message)
                    if data.get("audio"):
                        audio_chunk_count += 1
                        if audio_chunk_count % 5 == 0:
                            print(f"[{time.strftime('%H:%M:%S')}] 🎵 [ELEVENLABS] Đang gửi Audio xuống Client (Chunk {audio_chunk_count})...")
                        audio_bytes = base64.b64decode(data["audio"])
                        await client_websocket.send_bytes(audio_bytes)
                    if data.get("isFinal"):
                        print(f"[{time.strftime('%H:%M:%S')}] 🎵 [ELEVENLABS] Đã gửi XONG toàn bộ Audio gói này!")
                        break

            await asyncio.gather(sender(), receiver())
    except Exception as e:
        print(f"\n[{time.strftime('%H:%M:%S')}] ❌ [ELEVENLABS-LỖI]: {e}")

@app.websocket("/ws")
async def websocket_endpoint(client_websocket: WebSocket):
    await client_websocket.accept()
    print(f"\n{'='*50}\n[{time.strftime('%H:%M:%S')}] 🟢 [CLIENT] Web Frontend đã kết nối!\n{'='*50}")
    
    sys_prompt = get_system_prompt()
    session_state = {
        "final_transcript": "",
        "interim_transcript": "",
        "history": [{"role": "system", "content": sys_prompt}], 
        "last_audio_time": time.time(),
        "is_bot_working": False
    }
    
    dg_url = (
        "wss://api.deepgram.com/v1/listen?"
        "encoding=linear16&sample_rate=16000&channels=1"
        "&language=vi&model=nova-2&interim_results=true"
        "&utterance_end_ms=1000&vad_events=true"
        "&keepalive=true" 
    )
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    
    byte_count = 0
    try:
        dg_socket = await websockets.connect(dg_url, additional_headers=headers)
        print(f"[{time.strftime('%H:%M:%S')}] 🔗 [SERVER] Đã kết nối với Deepgram STT!")
        
        receiver_task = asyncio.create_task(deepgram_receiver(dg_socket, client_websocket, session_state))
        keepalive_task = asyncio.create_task(keep_deepgram_alive(dg_socket, client_websocket, session_state))

        while True:
            message = await client_websocket.receive()
            
            # XỬ LÝ AUDIO TỪ WEB LÊN DEEPGRAM
            if "bytes" in message:
                byte_count += 1
                if byte_count == 1:
                    print(f"\n[{time.strftime('%H:%M:%S')}] 🎤 [CLIENT->SERVER] Bắt đầu nhận luồng âm thanh PCM...")
                elif byte_count % 100 == 0:
                    print(".", end="", flush=True) 
                
                try:
                    await dg_socket.send(message["bytes"])
                except websockets.exceptions.ConnectionClosed:
                    print(f"\n[{time.strftime('%H:%M:%S')}] ⚠️ [DEEPGRAM] Kết nối gửi Audio bị đóng!")
                    break

            # XỬ LÝ ĐIỀU KHIỂN & GỌI LLM
            elif "text" in message:
                data = json.loads(message["text"])
                print(f"\n[{time.strftime('%H:%M:%S')}] 📩 [CLIENT->SERVER] Tín hiệu JSON: {data}")

                if data.get("type") == "control" and data.get("event") == "end_of_speech":
                    byte_count = 0 
                    session_state["is_bot_working"] = True
                    await client_websocket.send_json({"type": "status", "state": "thinking"})
                    
                    user_text = (session_state["final_transcript"] + session_state["interim_transcript"]).strip()
                    session_state["final_transcript"] = ""
                    session_state["interim_transcript"] = ""
                    
                    if not user_text:
                        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ [LLM] Không có text. Hủy bỏ!")
                        session_state["is_bot_working"] = False
                        session_state["last_audio_time"] = time.time()
                        await client_websocket.send_json({"type": "status", "state": "idle"})
                        continue
                    
                    print(f"[{time.strftime('%H:%M:%S')}] 🧠 [LLM] Chuẩn bị gửi câu hỏi tới OpenAI: '{user_text}'")
                    messages = session_state["history"] + [{"role": "user", "content": user_text}]
                    
                    try:
                        llm_model = "gpt-4o-mini" # Sử dụng model tiêu chuẩn để tránh lỗi API
                        
                        stream = await openai_client.chat.completions.create(
                            model=llm_model, 
                            messages=messages,
                            stream=True
                        )
                        
                        await client_websocket.send_json({"type": "status", "state": "speaking"})
                        print(f"[{time.strftime('%H:%M:%S')}] 🤖 [LLM-TEXT]: ", end="", flush=True)
                        
                        tts_queue = asyncio.Queue()
                        tts_task = asyncio.create_task(elevenlabs_tts_worker(tts_queue, client_websocket))
                        
                        llm_response = ""
                        buffer = ""
                        
                        async for chunk in stream:
                            content = chunk.choices[0].delta.content
                            if content:
                                llm_response += content
                                buffer += content
                                print(content, end="", flush=True) 
                                await client_websocket.send_json({"type": "llm_chunk", "text": content})
                                
                                if any(punct in buffer for punct in ['.', ',', '?', '!', '\n']):
                                    await tts_queue.put(buffer)
                                    buffer = ""
                        
                        if buffer:
                            await tts_queue.put(buffer)
                            
                        await tts_queue.put(None)
                        print(f"\n[{time.strftime('%H:%M:%S')}] 🧠 [LLM] Sinh text hoàn tất! Đang đợi ElevenLabs...")
                        await tts_task
                        
                        print(f"[{time.strftime('%H:%M:%S')}] ✅ [SERVER] Đã hoàn thành toàn bộ Pipeline cho câu hỏi này!")
                        await client_websocket.send_json({"type": "status", "state": "speaking_done"})
                        
                        session_state["history"].append({"role": "user", "content": user_text})
                        session_state["history"].append({"role": "assistant", "content": llm_response})
                        
                        if len(session_state["history"]) > 11:
                            session_state["history"] = [session_state["history"][0]] + session_state["history"][-10:]
                            
                    except Exception as llm_err:
                        print(f"\n[{time.strftime('%H:%M:%S')}] ❌ [LLM/OPENAI LỖI]: {llm_err}")
                        session_state["is_bot_working"] = False
                        session_state["last_audio_time"] = time.time()
                        await client_websocket.send_json({"type": "status", "state": "idle"})
                
                elif data.get("type") == "control" and data.get("event") == "playback_completed":
                    session_state["is_bot_working"] = False
                    session_state["last_audio_time"] = time.time()
                    await client_websocket.send_json({"type": "status", "state": "idle"})
                    print(f"[{time.strftime('%H:%M:%S')}] 🔊 [CLIENT] Frontend đã phát xong Audio. Chuyển về IDLE.")

    except WebSocketDisconnect:
        print(f"\n[{time.strftime('%H:%M:%S')}] 🔴 [CLIENT] Đã chủ động ngắt kết nối WebSocket!")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ [SERVER LỖI TỔNG]: {e}")
    finally:
        if 'receiver_task' in locals():
            receiver_task.cancel()
        if 'keepalive_task' in locals():
            keepalive_task.cancel()
        if 'dg_socket' in locals():
            try: await dg_socket.close()
            except Exception: pass

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8030))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
