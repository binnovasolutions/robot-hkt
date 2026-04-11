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
# (Lưu ý: ElevenLabs không hỗ trợ tham số "speed" qua API. 
# Tốc độ đọc phụ thuộc vào Voice ID và dấu câu)
# ==========================================
ELEVENLABS_MODEL = "eleven_flash_v2_5" 
ELEVENLABS_STABILITY = 0.35      # Hạ xuống 0.35 để giọng có cảm xúc, lên xuống tự nhiên hơn
ELEVENLABS_SIMILARITY = 0.85     # Tăng nhẹ để bám sát bản sao giọng gốc
ELEVENLABS_STYLE = 0.4           # Thêm Style (0.0 đến 1.0) để phóng đại biểu cảm, giúp giọng tươi vui hơn
ELEVENLABS_SPEAKER_BOOST = True  # Kích hoạt để làm rõ và trong giọng đọc
# ==========================================

if not DEEPGRAM_API_KEY or not OPENAI_API_KEY or not ELEVENLABS_API_KEY:
    print(f"[{time.strftime('%H:%M:%S')}] ⚠️ LỖI: Thiếu API Key trong file .env")

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
            await asyncio.sleep(1) # Kiểm tra mỗi giây
            
            # TRƯỜNG HỢP 1: Bot đang suy nghĩ/nói -> Bỏ qua đếm ngược idle
            # Đã bao gồm cả thời gian Frontend đang phát âm thanh qua loa
            if session_state.get("is_bot_working", False):
                bot_working_keepalive_timer += 1
                if bot_working_keepalive_timer >= 5:
                    await dg_socket.send(json.dumps({"type": "KeepAlive"}))
                    bot_working_keepalive_timer = 0
                
                session_state["last_audio_time"] = time.time()
                keepalive_count = 0
                continue

            # TRƯỜNG HỢP 2: Đếm ngược im lặng thực sự
            bot_working_keepalive_timer = 0
            idle_time = time.time() - session_state.get("last_audio_time", time.time())
            
            if idle_time >= 20:
                print(f"\n[{time.strftime('%H:%M:%S')}] ⏱️ Đã quá 20s không có tiếng nói. Backend chủ động đóng phiên làm việc!")
                try:
                    # Gửi tín hiệu để Frontend tự tắt Mic và đóng kết nối
                    await client_websocket.send_json({"type": "control", "event": "deepgram_timeout"})
                    # Đóng Deepgram
                    await dg_socket.close()
                    # Cấp cho Frontend 0.5s để nó tự đóng kết nối, nếu nó không đóng thì Backend ép đóng
                    await asyncio.sleep(0.5) 
                    await client_websocket.close()
                except Exception:
                    pass
                break # Thoát luồng

            elif idle_time >= 10 and keepalive_count == 1:
                await dg_socket.send(json.dumps({"type": "KeepAlive"}))
                print(f"[{time.strftime('%H:%M:%S')}] 📡 Đã gửi tín hiệu KeepAlive lần 2 (Chuẩn bị đóng nếu im lặng)...")
                keepalive_count = 2
            elif idle_time >= 5 and keepalive_count == 0:
                await dg_socket.send(json.dumps({"type": "KeepAlive"}))
                print(f"[{time.strftime('%H:%M:%S')}] 📡 Đã gửi tín hiệu KeepAlive lần 1 (Giữ kết nối Deepgram)...")
                keepalive_count = 1
            elif idle_time < 5:
                keepalive_count = 0

    except asyncio.CancelledError:
        pass
    except Exception as e:
        pass 

async def deepgram_receiver(dg_socket, client_websocket, session_state):
    """Luồng chạy ngầm nhận text từ Deepgram và gửi về Web"""
    try:
        async for message in dg_socket:
            res = json.loads(message)
            if isinstance(res, dict) and "channel" in res:
                channel = res.get("channel", {})
                if isinstance(channel, dict):
                    alternatives = channel.get("alternatives", [])
                    if isinstance(alternatives, list) and len(alternatives) > 0:
                        sentence = alternatives[0].get("transcript", "")
                        is_final = res.get("is_final", False)
                        
                        if sentence:
                            # Chỉ reset đồng hồ khi Deepgram THỰC SỰ nghe thấy chữ
                            session_state["last_audio_time"] = time.time()
                            
                            if is_final:
                                print(f"[{time.strftime('%H:%M:%S')}] ✅ Deepgram (Chốt câu): {sentence}")
                                session_state["final_transcript"] += sentence + " "
                                session_state["interim_transcript"] = "" 
                            else:
                                session_state["interim_transcript"] = sentence

                            await client_websocket.send_text(json.dumps({
                                "type": "transcript",
                                "text": sentence,
                                "is_final": is_final
                            }))
    except websockets.exceptions.ConnectionClosed:
        pass
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ⚠️ Lỗi luồng nhận Deepgram: {e}")

async def elevenlabs_tts_worker(tts_queue, client_websocket):
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}/stream-input?model_id={ELEVENLABS_MODEL}&output_format=pcm_16000"
    try:
        async with websockets.connect(uri) as el_socket:
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
                        await el_socket.send(json.dumps({"text": ""}))
                        break
                    if text_chunk.strip():
                        await el_socket.send(json.dumps({
                            "text": text_chunk, 
                            "try_trigger_generation": True
                        }))

            async def receiver():
                async for message in el_socket:
                    data = json.loads(message)
                    if data.get("audio"):
                        audio_bytes = base64.b64decode(data["audio"])
                        await client_websocket.send_bytes(audio_bytes)
                    if data.get("isFinal"):
                        break

            await asyncio.gather(sender(), receiver())
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Lỗi ElevenLabs TTS: {e}")

@app.websocket("/ws")
async def websocket_endpoint(client_websocket: WebSocket):
    await client_websocket.accept()
    print(f"[{time.strftime('%H:%M:%S')}] 🟢 Client đã kết nối!")
    
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
    
    try:
        dg_socket = await websockets.connect(dg_url, additional_headers=headers)
        print(f"[{time.strftime('%H:%M:%S')}] 🔗 Đã kết nối với Deepgram STT!")
        
        receiver_task = asyncio.create_task(deepgram_receiver(dg_socket, client_websocket, session_state))
        keepalive_task = asyncio.create_task(keep_deepgram_alive(dg_socket, client_websocket, session_state))

        while True:
            message = await client_websocket.receive()
            
            # XỬ LÝ AUDIO TỪ WEB LÊN DEEPGRAM
            if "bytes" in message:
                try:
                    await dg_socket.send(message["bytes"])
                except websockets.exceptions.ConnectionClosed:
                    break

            # XỬ LÝ ĐIỀU KHIỂN & GỌI LLM
            elif "text" in message:
                data = json.loads(message["text"])

                # --- 1. NGƯỜI DÙNG NÓI XONG, YÊU CẦU TRẢ LỜI ---
                if data.get("type") == "control" and data.get("event") == "end_of_speech":
                    session_state["is_bot_working"] = True # Bắt đầu khóa bộ đếm idle
                    await client_websocket.send_json({"type": "status", "state": "thinking"})
                    
                    user_text = (session_state["final_transcript"] + session_state["interim_transcript"]).strip()
                    session_state["final_transcript"] = ""
                    session_state["interim_transcript"] = ""
                    
                    if not user_text:
                        session_state["is_bot_working"] = False
                        session_state["last_audio_time"] = time.time()
                        await client_websocket.send_json({"type": "status", "state": "idle"})
                        continue
                    
                    print(f"\n[{time.strftime('%H:%M:%S')}] 🧠 Hỏi LLM: {user_text}")
                    messages = session_state["history"] + [{"role": "user", "content": user_text}]
                    
                    try:
                        stream = await openai_client.chat.completions.create(
                            model="gpt-5.4-nano", 
                            messages=messages,
                            stream=True
                        )
                        
                        await client_websocket.send_json({"type": "status", "state": "speaking"})
                        print(f"[{time.strftime('%H:%M:%S')}] 🤖 Trả lời: ", end="", flush=True)
                        
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
                        await tts_task
                        
                        print(f"\n[{time.strftime('%H:%M:%S')}] ✅ Server đã tạo xong luồng Audio!")
                        
                        # Chỉ báo cho Frontend là Audio đã tải xong. KHÔNG set idle ở đây nữa.
                        await client_websocket.send_json({"type": "status", "state": "speaking_done"})
                        
                        # Cập nhật lịch sử
                        session_state["history"].append({"role": "user", "content": user_text})
                        session_state["history"].append({"role": "assistant", "content": llm_response})
                        
                        if len(session_state["history"]) > 11:
                            session_state["history"] = [session_state["history"][0]] + session_state["history"][-10:]
                            print(f"[{time.strftime('%H:%M:%S')}] 🧹 Đã dọn dẹp bộ nhớ: Giữ lại System Prompt và 10 tin nhắn gần nhất.")
                            
                    except Exception as llm_err:
                        print(f"\n[{time.strftime('%H:%M:%S')}] ❌ Lỗi OpenAI/TTS: {llm_err}")
                        # Nếu lỗi thì xả cờ idle luôn để không bị treo
                        session_state["is_bot_working"] = False
                        session_state["last_audio_time"] = time.time()
                        await client_websocket.send_json({"type": "status", "state": "idle"})
                
                # --- 2. FRONTEND BÁO CÁO: ĐÃ PHÁT XONG ÂM THANH QUA LOA ---
                elif data.get("type") == "control" and data.get("event") == "playback_completed":
                    session_state["is_bot_working"] = False
                    session_state["last_audio_time"] = time.time() # Bắt đầu đếm 20s TỪ ĐÂY
                    await client_websocket.send_json({"type": "status", "state": "idle"})
                    print(f"[{time.strftime('%H:%M:%S')}] 🔊 Frontend đã phát xong Audio. Bắt đầu đếm giờ chờ...")

    except WebSocketDisconnect:
        print(f"\n[{time.strftime('%H:%M:%S')}] 🔴 Client đã chủ động ngắt kết nối!")
    except RuntimeError as e:
        if "Cannot call" in str(e):
            print(f"[{time.strftime('%H:%M:%S')}] 🔌 Đã đóng luồng WebSocket an toàn.")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ❌ Lỗi Runtime WebSocket: {e}")
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] ❌ Lỗi WebSocket: {e}")
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
    import os
    # Lấy cổng từ biến môi trường của hệ thống, mặc định là 8030 nếu không tìm thấy
    port = int(os.getenv("PORT", 8030))
    uvicorn.run("main:app", host="0.0.0.0", port=port)