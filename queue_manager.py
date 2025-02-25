import pika
import json
from config import RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_QUEUE

class QueueManager:
    """
    Manages a RabbitMQ queue for sending and receiving messages asynchronously.
    Useful for larger-scale or multi-user scenarios, but optional for single-user local mode.
    """

    def __init__(self):
        self.queue_name = RABBITMQ_QUEUE
        connection_params = pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT)
        self.connection = pika.BlockingConnection(connection_params)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=self.queue_name)
        print(f"📦 Queue '{self.queue_name}' initialized on {RABBITMQ_HOST}:{RABBITMQ_PORT}.")

    def send_message(self, message):
        """
        Sends a message to the RabbitMQ queue.
        'message' should be a Python dictionary, which we'll convert to JSON.
        """
        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue_name,
            body=json.dumps(message)
        )
        print(f"📤 Message sent to queue: {message}")

    def receive_message(self, callback):
        """
        Starts consuming messages from the RabbitMQ queue.
        For each received message, the provided callback is called.
        """

        def on_message(ch, method, properties, body):
            message = json.loads(body)
            print(f"📥 Message received: {message}")
            callback(message)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.queue_name, on_message_callback=on_message)
        print("🚀 Waiting for messages. Press CTRL+C to exit.")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            self.close_connection()

    def close_connection(self):
        """
        Closes the RabbitMQ connection gracefully.
        """
        if self.connection:
            self.connection.close()
            print("🔒 RabbitMQ connection closed.")